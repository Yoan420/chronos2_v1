# NYX for hourly day ahead electricity price forecasting across four European markets

Yoan Kesraoui

Correspondence: kesraoui.yoan@gmail.com

Research manuscript · 16 September 2026

## Abstract

Pretrained time series models can incorporate market fundamentals, but their use in electricity price forecasting requires evaluation beyond average point accuracy. This study examines NYX, a hybrid pipeline combining a frozen Chronos-2 model, a rolling CatBoost residual correction and a governed state space correction. A retrospective prequential evaluation covers Belgium, Germany, France and the Netherlands from 16 September 2025 to 15 September 2026. On 35,036 jointly observed country–hour pairs, NYX achieves a mean absolute error of 11.386 EUR/MWh, compared with 12.423 for the Chronos-2 stage and 11.482 after CatBoost correction, an 8.35% reduction relative to the foundation model stage. A separate industrial reporting composite achieves 11.309 EUR/MWh, with benchmark vintage limitations preventing a strictly prospective comparison. Nominal 80% prediction intervals cover only 72.99–75.87% of observations across countries, with substantially lower coverage during high-price events. Although the leading principal component explains 65.70% of simultaneous residual variance in an initial 180-day sample, a subsequent residual-learning experiment selects no additional correction on 95 validation days. None of the tested corrections improves mean absolute error over NYX on the reserved 90-day segment. These findings support the value of supervised residual adaptation within this pipeline while exposing unresolved tail risk and interval calibration. The evidence is conditional on reconstructed forecasts and available data snapshots; it does not establish prospective trading value or state-of-the-art performance.

Keywords: electricity price forecasting; time series foundation models; residual learning; rolling evaluation; prediction intervals; European electricity markets

## Practitioner summary

NYX improves average hourly price accuracy over its uncorrected foundation model in four European markets. Most of the observed improvement comes from the supervised residual stage; the additional governed filter provides a smaller gain. Nevertheless, large positive price spikes remain difficult and the reported prediction intervals are too narrow or misplaced to achieve their nominal coverage. Correlated errors across neighbouring markets do not by themselves justify another cross-market correction. For operational use, the priority is to verify historical information availability and recalibrate uncertainty before relying on the model for decisions sensitive to extreme prices. The results concern hourly prices and hourly aggregates, including the period after the transition to 15-minute market clearing.

## 1 Introduction

Electricity price forecasts inform scheduling, procurement and trading decisions, but their quality depends on the loss function and market conditions being considered. A forecast with modest average absolute error can still fail during rare price spikes. A forecast that reduces average bias can simultaneously increase absolute error. These distinctions are particularly relevant when a general-purpose time series model is combined with market-specific adaptation and then presented as an operational forecasting system.

Rigorous electricity price forecasting research requires comparisons on common observations, strong reference models, sufficiently broad evaluation periods and explicit treatment of statistical uncertainty. Lago et al. (2021) document how inconsistent datasets and weak evaluation practices complicate comparisons between proposed methods. Their recommendations motivate the present separation of model-stage comparisons, external benchmark comparisons and a distinct validation-based experiment. The objective is to characterize one identifiable NYX pipeline using its retained forecast artifacts, rather than to infer performance from demonstrations or selected favourable days.

Chronos-2 provides pretrained univariate, multivariate and covariate-informed forecasting capabilities through in-context information sharing (Ansari et al., 2025). Such a model can supply a transferable forecasting representation, while a supervised residual model adapts its systematic errors to a specific market. However, the combined system is no longer a purely zero-shot forecaster. Moreover, adding a correction to every forecast quantile changes location without necessarily improving the calibration of the underlying predictive distribution. Both point and interval evaluation are therefore needed.

Extreme-event classification provides a complementary perspective. Ma et al. (2026) combine adaptive thresholds, weighted XGBoost and SHAP analysis to forecast extreme-price occurrences. The present study instead evaluates continuous hourly price forecasts and quantiles. Realized extreme-price regimes are used to diagnose where errors occur; they are not treated as information available to the model at forecast issuance. Classification accuracy, ROC curves and SHAP explanations from another task would therefore not be interchangeable with the evidence reported here.

The study addresses three empirical questions. First, how much do the two residual stages improve accuracy relative to the same pipeline's frozen foundation model forecasts? Second, are the resulting intervals reliable overall and during negative or high-price hours? Third, can the remaining cross-market error dependence be converted into an additional forecast improvement using simple causal residual models?

The contribution is an empirical analysis of a layered forecasting system across four markets, with matched-support comparisons and a reproducible separation between descriptive evidence and validation-based model selection. It includes useful negative findings: the external reporting benchmark remains competitive, nominal intervals under-cover, and the tested additional residual models fail to improve the selected primary loss. No new foundation architecture or universal superiority claim is made.

## 2 Data and methodology

### 2.1 Forecast target and evaluation sample

The target is the hourly day-ahead electricity price, in EUR/MWh, for Belgium (BE), Germany (DE), France (FR) and the Netherlands (NL). The retained NYX artifacts identify the delivery date 16 September 2026 and contain a preceding 365-day evaluation window, from 16 September 2025 to 15 September 2026 inclusive. The future delivery date is excluded. There are 8,760 physical hourly observations per country and 35,040 country–hour pairs in total. Delivery days contain 23, 24 or 25 hours as dictated by daylight-saving time; the sample contains one 23-hour day, one 25-hour day and 363 ordinary days.

Timestamps are aligned in UTC for scoring. Civil-hour diagnostics use the corresponding local European time zone. Both physical occurrences of the repeated autumn hour remain in the target sample. No target observation is interpolated. The external Storm reporting series lacks one observation per country at 00:00 UTC on 26 October 2025. Four-way comparisons between Chronos-2, the CatBoost-corrected stage, NYX and Storm therefore use exactly 8,759 hours per country, or 35,036 points. NYX interval diagnostics use all 35,040 points.

@table:data_description

The evaluation period crosses the transition of Single Day-Ahead Coupling to a 15-minute market time unit in October 2025 (NEMO Committee, 2025). EPEX SPOT documents a 60-minute index formed as the arithmetic average of the corresponding quarter-hour clearing prices (EPEX SPOT, 2025). The evaluated NYX configuration, however, reads an already hourly provider series. The retained evidence does not reconcile every post-transition value with the four native quarter-hour prices. Consequently, this paper evaluates the hourly price representation supplied by the source and does not establish either verified four-quarter aggregation or quarter-hour forecast accuracy.

### 2.2 Input variables and information timing

Each country's Chronos-2 task receives its own historical prices, calendar variables and six forecast covariates: residual load in France, Germany, Belgium, the Netherlands and Spain, and forecast nuclear generation in France. The physical covariates are expressed in GW. Spain is a covariate source, not an evaluated target market. The nuclear variable is forecast generation rather than realized output or an installed-capacity measure. The four price targets are evaluated separately, although they share the pretrained model and cross-country fundamental inputs.

@table:inputs

The intended forecast origin for delivery day D is 08:00 Paris time on D−1. The preparation code selects covariates using civil-time cutoffs and requires complete future covariate coverage. Historical day-ahead prices through the end of D−1 can in principle be known before their physical delivery, because they arise from an earlier auction. This economic timing does not imply that a subsequently downloaded historical price series is an exact reconstruction of its original publication vintage.

The retained artifacts are reconstructed prequential backtests. They are not an archive of 365 forecasts actually issued at their historical origins. Integrity checks validate archive identities, checksums, alignment and recorded temporal invariants, and report no target-day observations assimilated before their forecasts. These checks establish properties of the replay implementation. They do not establish original provider publication times: the source audits explicitly report that provider revision timestamps and production-grade point-in-time evidence are unavailable. Historical labels may also have been refreshed before the artifacts were frozen.

The frozen Chronos-2 checkpoint is an additional temporal limitation. Its paper was released in October 2025, after the start of the evaluation window (Ansari et al., 2025). The earlier portion of the replay must therefore be interpreted as retrospective evaluation of a later available model. The absence of pretraining overlap with the evaluated markets has not been established. This prevents interpreting the entire annual replay as a historically deployable zero-shot experiment.

Storm is retained as an industrial reporting comparator and is excluded from all NYX input features. Its archived reporting series combines a frozen day-ahead cache with a native curve recomputed where the cache is missing. The annual evaluation includes 480 such fallback hours in BE and DE and 504 in FR and NL. On the reserved 90-day segment, these counts are 96 and 120 respectively. These observations are explicitly retained in the main descriptive comparison and excluded in a sensitivity analysis. Even the cache-only subset does not independently prove availability at the strict historical 08:00 cutoff.

### 2.3 The NYX forecasting pipeline

NYX denotes the nuclear-forecast pipeline with a frozen Chronos-2 stage, a supervised residual correction and a governed state space correction. The evaluated configuration contains no LoRA fine-tuning and no Storm blending. Figure 1 distinguishes these three stages from the later residual experiment.

@figure:figure1_design|Figure 1. NYX stages and the separate residual-learning experiment. Both production-pipeline corrections shift all retained quantiles equally. The 180/95/90-day split applies to the additional residual experiment, not to the historical development or selection of the complete NYX pipeline.

For country c, delivery day D and physical delivery hour h, let the base quantile at level τ be denoted by q with superscript C. The notation suppresses the country and delivery indices when unambiguous. Chronos-2 is used with a context of 2,048 physical hours and the required delivery-day horizon. It requests nine deciles and the downstream pipeline retains quantiles 0.1, 0.5 and 0.9. The implementation calls the frozen pretrained checkpoint separately for each country with cross-learning across independent tasks disabled; this does not disable within-task covariate attention.

The CatBoost stage learns the residual target given by the observed price minus the base median. It uses historical prequential forecasts and eligible features, including base quantiles, calendar effects and covariate profiles. CatBoost is a gradient boosting method designed to handle categorical information and prediction-shift issues (Prokhorenkova et al., 2018); the present task uses it as a residual regressor with an absolute-error objective. The fitted model is refitted daily over the preceding 365 civil days, subject to a minimum of 720 valid rows. Early warm-up days without sufficient data retain the uncorrected forecast.

$$r_{c,D,h}=y_{c,D,h}-q^{C}_{c,D,h}(0.5) \tag{1}$$

$$a_{c,D,h}=\operatorname{clip}\bigl(g_{c,D}(x_{c,D,h}),-40,40\bigr) \tag{2}$$

$$q^{R}_{c,D,h}(\tau)=q^{C}_{c,D,h}(\tau)+a_{c,D,h} \tag{3}$$

The final correction applies a governed bank of state space filters to the residuals of this corrected forecast. Linear bias, harmonic, market-feature and scale candidates, together with an unscented scale candidate, are available. The linear candidates follow the filtering principle of Kalman (1960); the nonlinear candidate requires a distinct observation mapping. A rolling training window ends before D. The target day's prices are not assimilated into its forecast. No smoothing pass using future observations is applied.

The governor evaluates recent historical losses over at most 60 days, with a minimum of 14 days, and searches a correction weight between zero and one on a 0.05 grid. It requires an improvement of at least the larger of 0.05 EUR/MWh and 0.5% of the base MAE. Otherwise, the identity correction is retained. Each candidate's raw correction is clipped to ±20 EUR/MWh before application of its selected weight. Writing b for that final correction gives:

$$q^{N}_{c,D,h}(\tau)=q^{R}_{c,D,h}(\tau)+b_{c,D,h},\qquad |b_{c,D,h}|\leq20 \tag{4}$$

The filter's internal replay is not a fully nested validation experiment: standardization and observation-noise quantities are estimated on the full admissible rolling training window before the recent historical replay is assessed. The window excludes D, but its recent inner scoring segment is not independent of those preprocessing estimates. The resulting annual comparison is therefore described as a retrospective rolling evaluation, without a claim of globally untouched model selection.

Because both correction stages add the same value to all quantiles, they preserve quantile order and interval width. In particular,

$$q^{N}(0.9)-q^{N}(0.1)=q^{C}(0.9)-q^{C}(0.1). \tag{5}$$

This identity is verified in the retained outputs up to numerical precision. The pipeline can move an interval towards an observation, but it does not learn a new interval width. A nominal P10–P90 label therefore requires an empirical coverage assessment.

@table:configuration

### 2.4 Comparators and additional residual models

The main component comparison uses the three retained NYX stages on identical observations. These are sequential stage comparisons, not separately optimized model families, and they do not identify the causal importance of individual physical covariates. Storm is compared separately because its vintage history and forecasting mechanism differ. A seasonal-naïve predictor repeats the frozen price from the same civil hour seven delivery days earlier. Repeated lagged civil hours are averaged solely for this input feature; a missing lagged hour is excluded. With the first seven evaluation days unavailable as targets for that comparator and the remaining alignment exclusions, its common-support comparison contains 8,590 hours per country across 358 delivery days.

The additional residual experiment asks whether remaining NYX errors are predictable from their own past, including cross-market information. These learners predict NYX minus the frozen observed price, and their output is subtracted from the NYX median. The validation search includes 13 candidates: the identity correction; country and country-by-hour exponentially weighted averages with half-lives of 7 or 28 days; univariate and multivariate ridge models with penalties 10 or 100; and principal-component ridge models with ranks one or two and the same penalties. This is a limited diagnostic family rather than an exhaustive architecture search.

For delivery D, all residual features and model-fitting labels are restricted to days no later than D−2. The features are the residual at the same civil hour on D−2, a country-level exponentially weighted mean and a country-by-hour weighted mean. Ridge features use a fixed 14-day half-life. The univariate model has three features per country; the multivariate model receives all twelve. Principal components are fitted to the four standardized residual targets within each training window. Each of the three four-country feature blocks is projected onto the same retained components, yielding three times the selected rank in predictors for component-score forecasting. No contemporary residual or Storm forecast enters these models.

The first 180 days form an initial sample. The next 95 days, from 15 March to 17 June 2026, are used to select hyperparameters and the primary family by pooled MAE. Fitting and validation selection use frozen labels; the main reported evaluation scores use audited refreshed observations, with both label versions checked in sensitivity analysis. The remaining 90 days, from 18 June to 15 September 2026, are evaluated after that selection is saved. Ridge models use up to 180 eligible days and are refitted every seven days; fresh day-ahead features are constructed daily. A 14-day feature warm-up leaves 165 eligible days at the first validation refit. Imputation, normalization and principal components are estimated only from each admissible fitting window.

Earlier test-segment observations may subsequently enter fitting windows under the fixed D−2 rule. Thus, the final segment evaluates an adapting prequential procedure with fixed hyperparameter choices, not a single model frozen for 90 days. The selection record and code ordering establish selection before test scoring within this experiment; they do not establish external preregistration or non-use of this period in earlier NYX development. All 151 recorded fitting events respect their label cutoff. A targeted EWMA feature check is invariant to perturbations of forbidden recent and future labels; this check is not a proof covering every operation of the complete pipeline.

### 2.5 Error measures and statistical uncertainty

For an aligned sample of n pairs, the signed error is the forecast minus the observation. Mean absolute error is the primary point-forecast measure; root mean squared error emphasizes large errors, and mean error measures bias. All three retain the unit EUR/MWh. Percentage losses are avoided because the target can be zero or negative.

$$\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|\widehat y_i-y_i|,\qquad \operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\widehat y_i-y_i)^2} \tag{6}$$

$$\operatorname{Bias}=\frac{1}{n}\sum_{i=1}^{n}(\widehat y_i-y_i). \tag{7}$$

Pooled scores are computed over the concatenated country–hour pairs. In the main full-support comparisons, country sample sizes coincide and pooled MAE is also the mean of country MAEs. Provenance sensitivities have unequal country counts and remain weighted by country–hour observations. Pooled RMSE is the square root of pooled squared error, rather than the arithmetic average of country RMSEs.

For lower and upper bounds l and u and median m, the central 80% interval score penalizes both width and observations outside the interval. The associated weighted interval score uses the median and this single interval (Bracher et al., 2021):

$$\operatorname{IS}_{0.2}=(u-l)+10\cdot\max(l-y,0)+10\cdot\max(y-u,0) \tag{8}$$

$$\operatorname{WIS}_{80}=\frac{0.5|y-m|+0.1\operatorname{IS}_{0.2}}{1.5}. \tag{9}$$

This single-interval WIS is not a full-distribution continuous ranked probability score. Lower values indicate better interval-and-median performance under the specified score. Empirical coverage and mean width are reported separately, since wide intervals can achieve coverage without being informative.

Negative-price diagnostics use realized prices below zero. High-price diagnostics use a country-specific 99th percentile estimated from frozen labels in the initial 180-day sample, then held fixed: 183.818 EUR/MWh in BE, 249.924 in DE, 158.461 in FR and 216.532 in NL. These are retrospective conditional diagnostics rather than ex ante event classifiers. In particular, high-price results for the full year include the sample used to estimate the threshold and are not an independent threshold-selection test. Exceedance rates need not remain at 1% as the price distribution changes.

Uncertainty is assessed with 2,000 paired moving-block bootstrap replicates using contiguous seven-day blocks, retaining the same resampled delivery days across all countries and compared models. Daily sums and counts reconstruct each pooled loss; both occurrences of daylight-saving hours remain included. Percentile 95% intervals follow the dependence-preserving principle of block resampling (Künsch, 1989). Annual stage intervals are post hoc descriptive analyses. Reserved-segment intervals are conditional on the selected hyperparameters and that summer-dominated segment. Neither analysis incorporates uncertainty in data vintages, prior pipeline selection or unobserved future regimes. Secondary comparisons have no multiplicity adjustment.

## 3 Experimental results

### 3.1 Annual point forecasting performance

NYX reduces MAE relative to its Chronos-2 stage in every country (Table 4). The pooled reduction is 1.037 EUR/MWh, from 12.423 to 11.386, or 8.35%. The paired annual seven-day block interval for NYX minus Chronos-2 is [−1.270, −0.760] EUR/MWh. Most of this improvement is already present after the CatBoost stage, whose pooled MAE is 11.482. The final governed filter reduces MAE by a further 0.095 EUR/MWh, with a descriptive interval [−0.169, −0.009]. These intervals describe the archived replay and do not correct for prior configuration selection.

@table:annual_stages

Relative to Storm's reporting composite, NYX reduces annual MAE only in Germany and has higher annual RMSE in all four countries. Its pooled MAE is 0.077 EUR/MWh higher, with a descriptive paired interval [−0.531, 0.759]. NYX's signed bias is closer to zero in all four countries. Thus, the evidence distinguishes improved centring from an improvement in the error distribution as a whole. The point estimates do not support a blanket assertion that NYX outperforms the industrial comparator.

@table:annual_benchmark

On the distinct weekly-naïve matched sample, pooled MAE is 34.000 EUR/MWh for the naïve predictor, 12.378 for Chronos-2, 11.480 for the supervised residual stage, 11.385 for NYX and 11.316 for Storm. This confirms that the pipeline improves substantially over a simple seasonal persistence reference on that support. It does not replace a comparison with strong separately fitted statistical and neural baselines. In particular, LEAR, independently tuned CatBoost price models and alternative foundation models have not been compared on this same four-country artifact set.

Figure 2 illustrates substantial temporal variation in performance. The first and final September bins are partial months, and each model is scored on exactly the same available hours within a bin. Such month-level differences are descriptive; the sample contains one annual cycle and cannot establish persistent seasonal superiority.

@figure:figure2_monthly_mae|Figure 2. Monthly MAE on common observations for the foundation stage, NYX and the Storm reporting composite. September 2025 and September 2026 are partial months. The external comparator contains recomputed fallback hours; its historical information set is not fully verified.

### 3.2 Extreme prices and interval reliability

The unconditional MAE hides a pronounced asymmetry. During high-price hours, NYX underpredicts on average in every country (Table 6). The German high-price subset has an MAE of 102.26 EUR/MWh and a bias of −87.46, despite an annual MAE near 11 EUR/MWh. The corresponding Dutch values are 73.23 and −55.05. France has many more exceedances of its fixed initial threshold but less severe average error, illustrating that exceedance counts and forecast difficulty are distinct quantities.

@table:extremes

The observed differences between countries should not be interpreted as estimates of a causal effect of generation mix or market coupling. These regime summaries condition on realized prices and use different country thresholds. They describe where the pipeline fails, while attribution to physical drivers would require additional controlled feature experiments or appropriately designed explanatory analyses.

Across all hours, nominal 80% intervals cover between 72.99% and 75.87% of observations (Table 7). Coverage during high-price hours falls to 33.65% in Germany and 52.26% in the Netherlands. Figure 3 locates more frequent interval misses among realized high-price observations. Because those subsets are selected using the realized target, neither their coverage nor their signed error establishes conditional miscalibration: even a calibrated predictive distribution can have both properties within an ex post upper-tail subset. These results motivate prospective event-risk evaluation, while undercoverage over the complete sample remains an observed marginal reliability limitation.

@table:intervals

@figure:figure3_interval_coverage|Figure 3. Empirical P10–P90 coverage over all hours and within realized negative- and high-price regimes. The dashed line is the nominal 80% coverage. High-price thresholds are fixed from the initial 180-day sample; conditional regime coverage is descriptive and need not equal unconditional nominal coverage even for a calibrated forecast.

The supervised correction reaches its absolute 40 EUR/MWh cap on only 0.11–0.50% of all country hours, but on 13.46% of the German high-price hours and 7.74% of the Dutch high-price hours. The final governed correction does not reach its 20 EUR/MWh cap during the evaluated year. This pattern identifies cap saturation as a relevant diagnostic; it does not demonstrate that increasing the cap would improve held-out accuracy. More fundamentally, fixed-width location corrections cannot directly respond to a state-dependent increase in forecast dispersion.

### 3.3 Residual dependence and further correction

On the initial 180-day sample, the first principal component explains 65.70% of the variance of the four simultaneous NYX residual series and the first two explain 85.89%. The participation ratio is 2.07 within the four-dimensional space. After country standardization, the first component still explains 63.65%. Residual correlations are strongest between BE and NL (0.772) and DE and NL (0.710), whereas FR has weaker associations with DE (0.180) and NL (0.247). Figure 4 summarizes this dependence.

@figure:figure4_residual_dependence|Figure 4. Simultaneous residual dependence in the initial 180 days. Panel (a) shows country correlations. Panel (b) uses eigenvalues of the unstandardized four-country covariance matrix. These are retrospective descriptions of contemporaneous errors, not forecasts of latent factors or evidence of exploitable advance information.

Validation selects the identity correction, retaining NYX as the primary model. Its pooled validation MAE is 12.315 EUR/MWh; all tested additional families have higher validation MAE. Table 8 reports the best validation-selected configuration from each family on the reserved segment. These secondary configurations are not chosen by their final-segment performance.

On the reserved 90 days, none of the five validation-selected correction configurations reduces MAE below NYX's 14.191 EUR/MWh. The country-level EWMA materially reduces the magnitude of signed bias, from −1.852 to −0.195, but increases MAE slightly. The principal-component ridge model has an MAE increase of 0.175 EUR/MWh, with a paired interval [0.026, 0.351]. Its RMSE point estimate improves by only 0.057 EUR/MWh, with interval [−0.254, 0.289], which does not establish an RMSE gain. Multivariate ridge performs worse than univariate ridge by the primary point estimate.

@table:residual_test

The result distinguishes simultaneous dependence from causal predictability. A low-dimensional representation can explain errors after their realization while failing to predict their signs and amplitudes at the next forecast origin. The experiment tests only the specified lagged-residual learners and losses. It does not exclude benefits from other admissible physical information, nonlinear methods or alternative adaptation strategies.

@figure:figure5_test_uncertainty|Figure 5. Reserved-segment MAE differences from NYX with paired 95% seven-day moving-block intervals. Positive values indicate larger loss than NYX. Secondary families are selected on validation; Storm is an external reporting comparison, not a candidate used to select the residual model. Intervals do not account for model-development or vintage uncertainty.

### 3.4 Sensitivity to observation and comparator provenance

Frozen and refreshed observations agree closely. After distinguishing float32 representation effects from substantive differences, there is one materially different hour per country, at 21:00 UTC on 16 June 2026. Absolute differences range from 6.955 to 9.305 EUR/MWh. The effect on annual MAE is approximately 0.001 EUR/MWh or less, and there is no material difference on the reserved segment. The cause of the discrepant versions is not identified here; numerical agreement cannot substitute for evidence of historical publication timing.

The refreshed observed-price snapshot also includes an EPEX fallback for delivery day 13 September 2026 in DE, FR and NL, totalling 72 evaluated country–hours. It was validated against overlapping source observations by the local reporting process. Table 9 excludes both recomputed Storm hours and, in a second sensitivity, these alternative-source labels. The descriptive ranking of pooled MAE remains unchanged. The sensitivity reduces reliance on these fallbacks, while leaving the broader historical-vintage limitation unresolved.

@table:provenance_sensitivity

The numerical audit independently reproduces all 70 reported reserved-segment scores and 70 bootstrap rows to floating-point precision. It also recomputes the annual stage comparisons on one shared support, avoiding a subtle discrepancy in which separately tabulated foundation and residual metrics originally retained one more hour than the Storm comparison. Reproducibility of these calculations is necessary for the empirical conclusions, but remains distinct from end-to-end prospective validation.

## 4 Discussion

### 4.1 What the hybrid structure contributes

The evidence supports a practical role for supervised residual learning around a frozen foundation model. Within the retained NYX pipeline, the CatBoost stage delivers most of the annual MAE improvement, and the governed state space stage contributes a smaller additional reduction. The result is consistent with systematic errors remaining after covariate-informed foundation inference. It does not establish that this combination is preferable to training a market-specific model directly, because that independent baseline has not been fitted on the same sample and information set.

The additional residual experiment places a useful constraint on further complexity. Even after strong contemporaneous cross-country dependence is identified, the tested adaptive corrections fail to improve the selected primary loss. The zero-correction option is consequently a necessary reference. Its selection prevents a diagnostic observation about correlation from being converted into an unsupported claim about forecast improvement.

The supervised stage, filter and additional ridge experiments pursue different fitting criteria and operate at different layers. In particular, minimizing squared residual error in ridge fitting does not guarantee a lower evaluation MAE. The bias reduction achieved by EWMA also shows why a single signed-error measure is inadequate for selecting a model intended to predict hourly price levels.

### 4.2 Reliability and operational interpretation

Interval undercoverage is an observed limitation of the current system. Recentring the same base quantiles can reduce point error while retaining a dispersion model that is inappropriate for some market states. An operational probability statement should therefore depend on empirical recalibration and subsequent evaluation of interval scores, coverage and width. Increasing width mechanically would not, by itself, establish a better probabilistic forecast.

Large high-price errors and the greater weight they receive under RMSE explain why annual MAE gains can coexist with weaknesses under other criteria. The appropriate operating choice depends on the decision problem and its costs. This study does not evaluate executable bids, positions, risk constraints, transaction costs or realized profit and loss. It therefore reports forecasting performance rather than economic value. The same distinction applies to the hourly aggregation: good hourly forecasts cannot establish skill for subhourly trading decisions.

The current diagnostic decomposition is transparent at the level of model stages, error regimes and correction magnitudes. It is not a SHAP analysis or a causal explanation of market drivers. The interpretability framework of Ma et al. (2026) could motivate a separate analysis, but its explanations and event-classification results cannot be transferred to NYX without running and validating the corresponding experiment.

### 4.3 Scope and threats to validity

The largest unresolved issue is the historical information set. Covariate cutoffs are enforced in the replay, yet provider revision timestamps are absent and production point-in-time proof is explicitly unavailable. The model checkpoint also postdates the beginning of the evaluated period. A prospective forecast archive or a replay using independently timestamped vintages is needed before interpreting the figures as attainable historical operational performance.

A second limitation is selection. The complete pipeline was developed before this manuscript and the annual sample is not established as an untouched holdout for that development. The extra 180/95/90-day experiment controls its own candidate selection, but cannot retroactively remove earlier choices from the NYX baseline. Annual block intervals describe sampling variation conditional on the retained forecasts; they do not provide a selection-adjusted test of a new architecture.

Third, the markets are geographically and economically related, and the sample covers one annual cycle. The reserved segment is dominated by summer conditions. Shared resampled blocks preserve cross-country dependence, but they do not create independent markets or additional years. The seven-day block length is a stated analysis choice rather than a demonstrated universal dependence horizon, and the numerous country and regime comparisons are exploratory.

Fourth, the comparison set is incomplete. The seasonal-naïve model is a basic reference and Storm has heterogeneous provenance. Strong independently trained baselines on a common data vintage, as advocated by Lago et al. (2021), remain necessary for a competitive benchmarking claim. Controlled input ablations would also be needed to quantify the contribution of nuclear and cross-country residual-load forecasts separately.

Finally, full public replication depends on access rights to the underlying market and provider data and the model artifacts. The local evidence package preserves aggregate results, analysis provenance and the scripts used for the manuscript; it does not establish that licensed data may be redistributed. These constraints limit the reproducibility claim to the available retained inputs and distinguish the present research manuscript from a completed public benchmark release.

## 5 Conclusion

This study evaluates an identifiable NYX pipeline for hourly day-ahead electricity prices in four European markets. In the retained annual replay, supervised residual adaptation and a governed filter reduce pooled MAE from 12.423 EUR/MWh for the Chronos-2 stage to 11.386 for NYX. The industrial reporting comparator remains competitive, and neither annual mean accuracy nor reduced bias eliminates large errors during price spikes. Nominal 80% intervals under-cover across every country.

An additional validation-based experiment retains the unmodified NYX forecast despite substantial contemporaneous residual dependence. The tested lagged-residual corrections yield no further MAE improvement on the reserved segment. The empirical lesson is that cross-market error structure must be converted into admissible predictive information before it justifies an extra model layer. The next evidential requirements are historically verifiable input vintages, prospective evaluation, interval recalibration and stronger matched-information baselines. The current results support a carefully bounded case study of hybrid forecasting, rather than a claim of universal superiority or demonstrated trading profitability.

## Data and code availability

All reported numerical analyses use retained local NYX artifacts identified by the nuclear_forecast_v1 engine, the civil_pit_v2 protocol and the 16 September 2026 artifact date. The accompanying evidence package contains aggregate CSV tables, manuscript figures, source scripts and checksums. Its reproduction instructions identify the required local forecast matrix and archive inputs. Raw provider data and model weights are not publicly deposited with this manuscript, and redistribution permissions have not been established. Public access conditions and an archival repository identifier should be finalized before submission.

## Author declarations

The author list and institutional affiliations, funding statement, competing-interest declaration and permissions to publish provider-derived results require confirmation by the author before journal submission. This draft does not assert institutional endorsement, funding status or absence of competing interests. AI assistance was used to prepare text, analysis code and document formatting; responsibility for scientific verification and the final submitted manuscript remains with the author. No new NYX training run or prospective market experiment was performed in preparing this manuscript.

## References

Ansari, A. F., Shchur, O., Küken, J., et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. [https://doi.org/10.48550/arXiv.2510.15821](https://doi.org/10.48550/arXiv.2510.15821)

Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating epidemic forecasts in an interval format. PLOS Computational Biology, 17(2), e1008618. [https://doi.org/10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)

EPEX SPOT. (2025). 15-minute products in market coupling. Market design information and hourly price-index definition. Accessed 16 September 2026. [https://www.epexspot.com/en/new-15-minute-products-market-coupling](https://www.epexspot.com/en/new-15-minute-products-market-coupling)

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1), 35–45. [https://doi.org/10.1115/1.3662552](https://doi.org/10.1115/1.3662552)

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217–1241. [https://doi.org/10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)

Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy, 293, 116983. [https://doi.org/10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)

Ma, J., Chen, Y.-w., & Meng, F. (2026). An interpretable machine learning approach for forecasting the occurrence of extreme electricity prices in the day-ahead market. Journal of the Operational Research Society. Advance online publication. [https://doi.org/10.1080/01605682.2026.2660989](https://doi.org/10.1080/01605682.2026.2660989)

NEMO Committee. (2025). Single Day-Ahead Coupling: implementation of the 15-minute market time unit. Official communications dated 1 and 7 October 2025, listed on the SDAC resource page. Accessed 16 September 2026. [https://www.nemo-committee.eu/sdac](https://www.nemo-committee.eu/sdac)

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31. [https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)

## Appendix A Reproducibility details

The retained foundation checkpoint is amazon/chronos-2 at revision 29ec3766d36d6f73f0696f85560a422f50e8498c. The evaluated output median is the final residual_kalman quantile 0.5; the diagnostic stages are the corresponding Chronos-2 and residual_corrected medians from the same artifact. Identical names from another run or configuration are not interchangeable with these outputs.

The annual artifact includes a preceding warm-up and daily rolling refits. The first 180 evaluation days used by the additional residual experiment are therefore not the foundation model's pretraining sample or the only fitting history used by the complete NYX pipeline. This distinction is essential when reproducing the analysis.

The bootstrap uses daily country-specific sums of absolute and squared error and sample counts, draws shared contiguous seven-day blocks and truncates the concatenated draws to the required number of days. The additional residual experiment uses random seed 20260915. The separately computed annual comparisons use the seed recorded in the accompanying audit script. No hour is treated as independent merely to narrow an uncertainty interval.

All table figures are rounded only for display. Supporting CSV files retain full numerical precision. Positive bias means overprediction; in the reserved-segment table and forest plot, a positive MAE difference means worse performance than NYX. Annual stage comparisons explicitly report NYX minus the stated reference. The same distinction applies when reading the corresponding bootstrap tables.

Model configuration values and feature details are reconstructed from retained configuration files and the code audit. A recorded minimum-leaf parameter is not listed as an effective CatBoost setting because it is not forwarded by the actual estimator constructor. Historical software environment versions and hardware for full NYX training have not been comprehensively archived in this manuscript. Timings from the lightweight residual audit are therefore not presented as end-to-end NYX computational cost.

## Appendix B State space correction details

Let the upstream median be the CatBoost-corrected median, and let a denote the CatBoost shift. Candidate filter states are initialized afresh for each delivery D, replayed over the preceding 365 civil days and then used to forecast D. Each training day's forecast precedes that day's hourly state updates. The five candidate observation maps are listed in Table 10. Hourly sine and cosine terms use a 24-hour period.

@table:filters

For the market candidate, continuous regressors are centred by their training-window medians, scaled by the maximum of 1.4826 times the median absolute deviation, the population standard deviation and 10⁻⁶, and clipped to [−5, 5]. The shared observation-noise variance R is the larger of one and the squared robust scale of the upstream residuals. These quantities are estimated once on the whole admissible window, with the internal-selection limitation described in Section 2.3.

The linear candidates use an identity daily transition. Initial state means are zero except that the linear scale starts at one. Initial covariance is R times the identity, with scale-coordinate variance replaced by 0.05². Daily process variance is 0.001R per ordinary coordinate and 0.0001 × 0.05² for the scale coordinate. The unscented scale logit has daily persistence 0.99 and is mapped to a bounded scale through a logistic function. The archived filtering implementation records pykalman version 0.11.2.

For a linear observation vector x, the hourly gain and state update are:

$$K=\frac{Px}{x^{\mathsf T}Px+R},\qquad m^{+}=m+K\widetilde\nu,\qquad P^{+}=P-Kx^{\mathsf T}P. \tag{10}$$

Here the innovation is clipped to ±3 times the square root of R before updating the state, with additional numerical covariance safeguards. Bias, harmonic and market candidates observe the upstream residual. Scale candidates observe the error relative to the Chronos median and model the CatBoost shift multiplicatively, together with an additive bias. The unscented candidate applies its nonlinear observation map through the unscented filter. The resulting proposed additive correction is clipped before weighting:

$$b_{D,h}=w_D\operatorname{clip}(g_{D,h},-20,20),\qquad w_D\in\{0,0.05,\ldots,1\}. \tag{11}$$

The same selected candidate and weight apply to the whole delivery day. The configured confirmation window is zero; no independent confirmation segment is implied. This specification concerns the evaluated governed correction only and excludes separate extreme-price experts and other experimental NYX variants found elsewhere in the project.
