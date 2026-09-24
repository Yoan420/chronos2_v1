from pathlib import Path
import re,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
TASK=Path(__file__).resolve().parent
OUT=ROOT/'output/nyx_paper_20260831'
SRC=TASK/'analysis'
s=(ROOT/'output/nyx_paper_20260916/NYX_Research_Manuscript.md').read_text(encoding='utf-8')
def replace_par(prefix,new):
    global s
    blocks=s.split('\n\n')
    found=[i for i,b in enumerate(blocks) if b.startswith(prefix)]
    assert len(found)==1,(prefix,found)
    blocks[found[0]]=new
    s='\n\n'.join(blocks)

replace_par('Research manuscript','Research manuscript · Evaluation cutoff 31 August 2026')
replace_par('Pretrained time series models', 'Pretrained time series models can incorporate market fundamentals, but their use in electricity price forecasting requires evaluation beyond average point accuracy. This study examines NYX, a hybrid pipeline combining a frozen Chronos-2 model, a rolling CatBoost residual correction and a governed state space correction. A retrospective prequential evaluation covers Belgium, Germany, France and the Netherlands from 16 September 2025 to 31 August 2026. On 33,596 jointly observed country–hour pairs, NYX achieves a mean absolute error of 11.189 EUR/MWh, compared with 12.197 for the Chronos-2 stage and 11.284 after CatBoost correction, an 8.27% reduction relative to the foundation model stage. A separate industrial reporting composite achieves 11.162 EUR/MWh, with benchmark vintage limitations preventing a strictly prospective comparison. The mathematical formulation distinguishes pretrained quantile inference, supervised nonlinear error adaptation and bounded online state correction. Their shared additive action preserves interval width, and nominal 80% intervals cover only 73.23–76.24% of observations across countries. Although the leading principal component explains 65.70% of simultaneous residual variance in an initial 180-day sample, a subsequent residual-learning experiment selects no additional correction on 95 validation days. None of the tested corrections improves pooled mean absolute error over NYX on the reserved 75-day segment. The evidence is conditional on reconstructed forecasts, available data snapshots and implementation limitations; it does not establish prospective trading value or state-of-the-art performance.')
replace_par('The target is the hourly', 'The target is the hourly day-ahead electricity price, in EUR/MWh, for Belgium (BE), Germany (DE), France (FR) and the Netherlands (NL). The analysis retains delivery dates from 16 September 2025 through 31 August 2026 inclusive, excluding every later observation. This gives 350 civil days, 8,400 physical hourly observations per country and 33,600 country–hour pairs in total. Delivery days contain 23, 24 or 25 hours as dictated by daylight-saving time; the sample contains one 23-hour day, one 25-hour day and 348 ordinary days. The cutoff limits the evaluated delivery dates; it does not assert that the retrospectively assembled forecast and observation snapshots were already available on that date.')
replace_par('Storm is retained', 'Storm is retained as an industrial reporting comparator and is excluded from all NYX input features. Its archived reporting series combines a frozen day-ahead cache with a native curve recomputed where the cache is missing. The 350-day evaluation includes 456 such fallback hours in BE and DE and 480 in FR and NL. On the reserved 75-day segment, these counts are 72 and 96 respectively. These observations are retained in the main descriptive comparison and excluded in a sensitivity analysis. Even the cache-only subset does not independently prove availability at the strict historical 08:00 cutoff.')
replace_par('The main component comparison', 'The main component comparison uses the three retained NYX stages on identical observations. These are sequential stage comparisons, not separately optimized model families, and they do not identify the causal importance of individual physical covariates. Storm is compared separately because its vintage history and forecasting mechanism differ. A seasonal-naïve predictor repeats the frozen price from the same civil hour seven delivery days earlier. Repeated lagged civil hours are averaged solely for this input feature; a missing lagged hour is excluded. After the first seven delivery days and remaining alignment exclusions, the five-model common-support comparison contains 8,230 hours per country across 343 delivery days.')
replace_par('The first 180 days form', 'The first 180 days form an initial sample. The next 95 days, from 15 March to 17 June 2026, are used to select hyperparameters and the primary family by pooled MAE. Fitting and validation selection use frozen labels; the main reported evaluation scores use audited refreshed observations, with both label versions checked in sensitivity analysis. The final 75 retained days, from 18 June to 31 August 2026, are evaluated using the original saved choices. The cutoff revision neither reselects configurations nor retrains models. Ridge models use up to 180 eligible days and are refitted every seven days; fresh day-ahead features are constructed daily. A 14-day feature warm-up leaves 165 eligible days at the first validation refit. Imputation, normalization and principal components are estimated only from each admissible fitting window.')
replace_par('Earlier test-segment', 'Earlier test-segment observations may subsequently enter fitting windows under the fixed D−2 rule. Thus, the final segment evaluates an adapting prequential procedure with fixed hyperparameter choices, not a single model frozen for 75 days. The selection record and code ordering establish selection before test scoring within this experiment; they do not establish external preregistration or non-use of this period in earlier NYX development. All 145 fitting events contributing to the retained validation and test dates respect their label cutoff; the final test refit is on 27 August, using labels through 25 August. A targeted EWMA feature check is invariant to perturbations of forbidden recent and future labels; this check is not a proof covering every operation of the complete pipeline.')
replace_par('NYX reduces MAE relative', 'NYX reduces MAE relative to its Chronos-2 stage in every country (Table 4). The pooled reduction is 1.008 EUR/MWh, from 12.197 to 11.189, or 8.27%. The paired 350-day seven-day block interval for NYX minus Chronos-2 is [−1.240, −0.715] EUR/MWh. Most of this improvement is already present after the CatBoost stage, whose pooled MAE is 11.284. The final governed filter reduces MAE by a further 0.094 EUR/MWh, with a descriptive interval [−0.176, −0.007]. These intervals describe the archived replay and do not correct for prior configuration selection.')
replace_par("Relative to Storm's", "Relative to Storm's reporting composite, NYX reduces full-period MAE only in Germany and has higher full-period RMSE in all four countries. Its pooled MAE is 0.027 EUR/MWh higher, with a descriptive paired interval [−0.561, 0.736]. NYX's signed bias is closer to zero in all four countries. Thus, the evidence distinguishes improved centring from an improvement in the error distribution as a whole. The point estimates do not support a blanket assertion that NYX outperforms the industrial comparator.")
replace_par('On the distinct weekly-naïve', 'On the distinct weekly-naïve matched sample, pooled MAE is 33.645 EUR/MWh for the naïve predictor, 12.145 for Chronos-2, 11.278 for the supervised residual stage, 11.184 for NYX and 11.167 for Storm. This confirms that the pipeline improves substantially over a simple seasonal persistence reference on that support. It does not replace a comparison with strong separately fitted statistical and neural baselines. In particular, LEAR, independently tuned CatBoost price models and alternative foundation models have not been compared on this same four-country artifact set.')
replace_par('Figure 2 illustrates', 'Figure 2 illustrates substantial temporal variation in performance. September 2025 is a partial month; the final bin covers all of August 2026. Each model is scored on exactly the same available hours within a bin. Such month-level differences are descriptive; the sample covers slightly less than a year and cannot establish persistent seasonal superiority.')
replace_par('@figure:figure2_', '@figure:figure2_monthly_mae|Figure 2. Monthly MAE on common observations from 16 September 2025 through 31 August 2026. September 2025 is partial. The external comparator contains recomputed fallback hours; its historical information set is not fully verified.')
replace_par('The unconditional MAE hides', 'The unconditional MAE hides a pronounced asymmetry. During high-price hours, NYX underpredicts on average in every country (Table 6). The German high-price subset has an MAE of 109.51 EUR/MWh and a bias of −95.02, despite an overall MAE near 11 EUR/MWh. The corresponding Dutch values are 95.79 and −74.47. France has many more exceedances of its fixed initial threshold but less severe average error, illustrating that exceedance counts and forecast difficulty are distinct quantities.')
replace_par('Across all hours, nominal', 'Across all hours, nominal 80% intervals cover between 73.23% and 76.24% of observations (Table 7). Coverage during high-price hours falls to 29.41% in Germany and 43.81% in the Netherlands. Figure 3 locates more frequent interval misses among realized high-price observations. Because those subsets are selected using the realized target, neither their coverage nor their signed error establishes conditional miscalibration: even a calibrated predictive distribution can have both properties within an ex post upper-tail subset. These results motivate prospective event-risk evaluation, while undercoverage over the complete sample remains an observed marginal reliability limitation.')
sat=pd.read_csv(SRC/'correction_saturation.csv')
ext=sat[sat.grouping=='extreme']
# Extreme rows use the source's grouping labels, preserved in the supporting CSV.
f=pd.read_csv(SRC/'verified_hourly_pairs.csv.gz')
meta=json.loads((SRC/'cutoff_manifest.json').read_text())
caps={}
for z in ['DE','NL']:
    g=f[(f.zone==z)&(f.observed>meta['thresholds_initial_train_frozen_actual'][z]['q99'])]
    caps[z]=100*(abs(g.residual_correction)>=40-1e-8).mean()
replace_par('The supervised correction reaches',f"The supervised correction reaches its absolute 40 EUR/MWh cap on only 0.12–0.52% of each country's hours, but on {caps['DE']:.2f}% of the German high-price hours and {caps['NL']:.2f}% of the Dutch high-price hours. The final governed correction does not reach its 20 EUR/MWh cap during the evaluated period. This pattern identifies cap saturation as a relevant diagnostic; it does not demonstrate that increasing the cap would improve held-out accuracy. More fundamentally, fixed-width location corrections cannot directly respond to a state-dependent increase in forecast dispersion.")
replace_par('On the reserved 90 days', "On the reserved 75 days, none of the five validation-selected correction configurations reduces pooled MAE below NYX's 13.833 EUR/MWh. The country-level EWMA reduces the magnitude of signed bias, from −2.140 to −0.331, but raises pooled MAE to 13.861. The principal-component ridge model has an MAE increase of 0.237 EUR/MWh, with a paired interval [0.086, 0.427]. Its RMSE point estimate improves by only 0.036 EUR/MWh, with interval [−0.275, 0.359], which does not establish an RMSE gain. Multivariate ridge performs worse than univariate ridge by the primary point estimate. These pooled results do not imply that every individual-country point estimate is worse.")
replace_par('The refreshed observed-price snapshot', 'There are no alternative-source EPEX observation fallbacks within the retained dates. Table 9 therefore compares the complete matched sample with the subset excluding recomputed Storm values. The descriptive ranking of pooled MAE remains unchanged. This sensitivity reduces reliance on comparator fallbacks, while leaving the broader historical-vintage limitation unresolved.')
replace_par('The numerical audit independently', 'The cutoff-specific numerical audit recomputes all 70 reserved-segment score rows and 70 bootstrap rows from the saved forecasts, retaining both observation versions and the original validation-selected configurations. Stage comparisons use one shared support. No post-cutoff observations enter the revised statistics, resampled blocks or figures. Reproducibility of these calculations remains distinct from end-to-end prospective validation.')
replace_par('This study evaluates an identifiable', 'This study evaluates an identifiable NYX pipeline for hourly day-ahead electricity prices in four European markets through 31 August 2026. In the retained 350-day replay, supervised residual adaptation and a governed filter reduce pooled MAE from 12.197 EUR/MWh for the Chronos-2 stage to 11.189 for NYX. Their mathematical interaction combines frozen quantile representation, nonlinear error learning and bounded state adaptation without joint training. The industrial reporting comparator remains competitive, and neither mean accuracy nor reduced bias eliminates large errors during price spikes. Equal shifts preserve the original quantile widths, and nominal 80% intervals under-cover across every country.')
replace_par('All reported numerical analyses', 'All reported numerical analyses use retained local NYX artifacts identified by the nuclear_forecast_v1 engine and civil_pit_v2 protocol, restricted to delivery dates no later than 31 August 2026. The source snapshots were assembled retrospectively after that cutoff; their original dates and checksums are retained in the evidence manifest. The accompanying package contains aggregate CSV tables, manuscript figures, source scripts and checksums. Its reproduction instructions identify the required local forecast matrix and archive inputs. Raw provider data and model weights are not publicly deposited with this manuscript, and redistribution permissions have not been established. Public access conditions and an archival repository identifier should be finalized before submission.')
replace_par('The bootstrap uses daily', 'The bootstrap uses daily country-specific sums of absolute and squared error and sample counts, draws shared contiguous seven-day blocks and truncates the concatenated draws to the required number of days. All revised bootstrap calculations use 2,000 replicates and the fixed original seed 20260915; the integer is a reproducibility setting, not the evaluation cutoff. No hour is treated as independent merely to narrow an uncertainty interval.')

# Global changes affect period descriptions, never the rolling 365-day fitting windows.
s=s.replace('8,759','8,399').replace('35,036','33,596').replace('35,040','33,600')
s=s.replace('archive of 365 forecasts','archive of 350 forecasts')
s=s.replace('180/95/90','180/95/75').replace('reserved 90-day','reserved 75-day')
s=s.replace('Annual point forecasting performance','Point forecasting performance through August 2026')
s=s.replace('annual sample','350-day sample').replace('annual replay','350-day replay').replace('annual evaluation','350-day evaluation')
s=s.replace('annual stage','full-period stage').replace('Annual stage','Full-period stage').replace('Annual block','Full-period block').replace('annual block','full-period block')
s=s.replace('annual MAE','full-period MAE').replace('annual mean','full-period mean').replace('annual comparison','full-period comparison').replace('Annual comparison','Full-period comparison')
s=s.replace('retained annual replay','retained 350-day replay').replace('annual artifact','source artifact').replace('entire annual','entire 350-day')
s=s.replace('for the full year','for the full period').replace('covers one annual cycle','covers slightly less than a year')
s=s.replace('approximately 0.001 EUR/MWh or less','at most 0.00111 EUR/MWh in absolute value')
s=s.replace(' Accessed 16 September 2026.','')

# Expanded mathematical section, with a separate branch for the secondary models.
theory=(TASK/'theory_draft.md').read_text(encoding='utf-8')
start=theory.index('#### 2.3.1')
end=theory.index('# Evidence notes')
theory=theory[start:end].strip()
split=theory.index('A separate diagnostic learner')
core=theory[:split].strip()
extra=theory[split:].strip()
core=re.sub(r'^### 2\.4[^\n]*','',core,flags=re.M).strip()
core=core.replace('For each D, CatBoost fits past prediction errors using dates $\\mathcal W_D=\\{D-365,\\ldots,D-1\\}$.', 'For each country and delivery D, CatBoost fits past prediction errors on the hourly index set $\\mathcal W_D=\\{j:D-365\\leq\\operatorname{day}(j)<D\\}$, with $n_D=|\\mathcal W_D|$.')
core=core.replace('Let $c$ denote a country', 'The production corrections and governors are fitted separately for each country. We suppress the country index in their local equations; their states, noise estimates and daily weights remain country-specific. Cross-country dependence enters through the forecast covariates and the separately evaluated multivariate residual learners.\n\nLet $c$ denote a country')
core=core.replace('The next layer tracks residual structure remaining after CatBoost.', r'''The next layer tracks residual structure remaining after CatBoost. Its state space interpretation is

$$\xi_d=A\xi_{d-1}+\eta_d,\qquad z_{d,h}=\mathcal H(\xi_d,x_{d,h})+\varepsilon_{d,h}.\tag{STATE}$$

Here $\xi_d$ is a vector of latent correction coefficients, $\eta_d$ is daily process noise with covariance $Q$, and $\varepsilon_{d,h}$ is zero-mean working observation noise with variance $R$. In the linear case, $\mathcal H(\xi,x)=x^{\mathsf T}\xi$; the scale candidate uses the nonlinear map specified below. This is the filter's working model rather than an empirically verified generative law. Sequential conditioning updates the state moments after each training-hour label, while process noise is added at the daily transition.

''')
core=core.replace('Its market regressors', 'The market regressors')
core=core.replace('through $h(\\theta)=b_0+s(u)a_h$', 'through $\\mathcal H(\\xi)=b_0+s(u)a_h$, where $\\xi=(b_0,u)^{\\mathsf T}$')
core=core.replace('h(m)',r'\mathcal H(m)')
core=core.replace('pooled hourly MAE over up to 60 replay days', 'hourly MAE aggregated within the country over up to 60 replay days')
# The candidate matrix is easier to read than an overlong set-valued formula.
core=re.sub(r'The five candidate shifts.*?\\tag\{T13\}\$\$', 'The five candidate observation maps are listed in Table 10 in Appendix B. They include additive bias, daily harmonics, market-feature adaptation and linear or nonlinear rescaling of the CatBoost shift.',core,flags=re.S)
intro='''### 2.3 Mathematical formulation of the NYX pipeline

NYX denotes the nuclear-forecast pipeline with a frozen Chronos-2 stage, a supervised residual correction and a governed state space correction. The evaluated configuration contains no LoRA fine-tuning and no Storm blending. Figure 1 distinguishes the production cascade from the later residual experiment.

@figure:figure1_design|Figure 1. NYX stages and the separate residual-learning experiment through 31 August 2026. Both production-pipeline corrections shift all retained quantiles equally. The 180/95/75-day split applies only to the additional residual experiment.

'''
a=s.index('### 2.3 ');b=s.index('### 2.4 ')
s=s[:a]+intro+core+'\n\n@table:configuration\n\n'+s[b:]
# Insert detailed secondary theory before fixed experimental selection protocol.
pos=s.index('The first 180 days form')
s=s[:pos]+extra+'\n\n'+s[pos:]
# Replace duplicated secondary feature prose with a brief lead-in.
replace_par('For delivery D, all residual features', 'The following equations specify the secondary residual learners and clarify why a contemporaneous factor structure need not yield a forecast gain. These learners act after the complete NYX cascade and are evaluated separately from its internal governor.')

# Appendix is the readable candidate map and the exact numerical caveat.
s=s[:s.index('## Appendix B State space')]+'''## Appendix B State space candidates and numerical limitation

The upstream median is the CatBoost-corrected median and $a$ denotes the CatBoost shift. Each candidate state is reinitialized for delivery D and replayed over its admissible 365-day history. Table 10 specifies the correction relative to that upstream median; the observation equation used for updating scale candidates is instead the Chronos-relative error $b_0+sa$.

@table:filters

The optimized nonlinear candidate has a sigma-point orientation discrepancy. Let $L$ be the lower-triangular NumPy Cholesky factor, so that $P=LL^{\\mathsf T}$. The code builds its symmetric offsets from rows of $L$, whereas covariance-preserving sigma points require its columns. With the implemented weights, the represented covariance is $L^{\\mathsf T}L$, which generally differs from $P$. For example, $P=\\left[\\begin{matrix}4&1\\\\1&1\\end{matrix}\\right]$ gives represented covariance approximately $\\left[\\begin{matrix}4.25&0.433\\\\0.433&0.75\\end{matrix}\\right]$. This is a read-only analytic check, not a replacement forecast experiment.

Within the retained dates, this candidate is selected only for the Netherlands on 21 February 2026, comprising 24 hours at weight 0.45. It is never selected in the reserved 75-day segment. The small observed selection count does not bound the effect of a corrected implementation: altered candidate losses could change governance choices on other dates. The manuscript therefore reports the archived governed implementation, not a validated canonical UKF. No forecast has been silently repaired. A corrected replay and renewed evaluation are required before an accuracy claim is attributed to the standard unscented filter.
'''
# Refer to the limitation in the main discussion as well as the method.
needle='Finally, full public replication'
pos=s.index(needle)
s=s[:pos]+'An implementation limitation also affects the nonlinear filter candidate: its sigma-point construction does not in general reproduce the intended covariance. Appendix B gives the mathematical discrepancy and its observed selection dates. The reported forecasts are unchanged, so these results concern the archived implementation; a corrected rerun could alter both candidate forecasts and governor decisions.\n\n'+s[pos:]
# Renumber display equations in reading order, including the scoring definitions.
counter=[0]
def number(m):
    counter[0]+=1
    return r'\tag{'+str(counter[0])+'}'
s=re.sub(r'\\tag\{[^}]+\}',number,s)
s=s.replace('documented below','documented in Appendix B')
s=s.replace('is daily process noise with covariance', 'is zero-mean daily process noise with covariance')
s=s.replace('the scale candidate uses the nonlinear map', 'the unscented scale candidate uses the nonlinear map')
s=s.replace(' Each country has separately fitted residual and filter stages; suppressed country indices, including on governance weights, remain country-specific.', '')
s=s.replace('magnitude. (Prokhorenkova et al., 2018); (CatBoost developers, n.d.).', 'magnitude (Prokhorenkova et al., 2018; CatBoost developers, n.d.).')
s=s.replace('the observation equation used for updating scale candidates is instead the Chronos-relative error $b_0+sa$.', 'their predicted observation is $b_0+sa$ and their assimilated target is the Chronos-relative error $y-q^C(0.5)$.')
s=s.replace('All 145 fitting events', 'The retained test is a truncated subset of the previously examined test period, not a newly untouched holdout. All 145 fitting events')
# Use author-year citations in the article; keep implementation URLs in references.
links={
'Official normalization implementation':'Amazon Science contributors, n.d.',
'Official attention layers':'Amazon Science contributors, n.d.',
'Official output and loss implementation':'Amazon Science contributors, n.d.',
'Ansari et al., 2025, Sections 3.1–3.2':'Ansari et al., 2025',
'CatBoost paper':'Prokhorenkova et al., 2018',
'official MAE definition':'CatBoost developers, n.d.',
'Official parameter distinctions':'CatBoost developers, n.d.',
'Official Ridge objective':'Scikit-learn developers, n.d.'}
for label,cite in links.items():
    s=re.sub(r'\['+re.escape(label)+r'\]\(https?://[^)]+\)', '('+cite+')',s)
s=re.sub(r'\. \((Amazon Science contributors|Ansari et al\.|CatBoost developers|Scikit-learn developers|Prokhorenkova et al\.)([^)]*)\)\.',r' (\1\2).',s)
s=s.replace('The nonlinear candidate propagates five sigma points', 'Unscented filtering approximates nonlinear transformations by propagating deterministic sigma points (Julier and Uhlmann, 2004). The archived nonlinear candidate propagates five sigma points')
s=s.replace('For a linear measurement vector', 'Following the linear filtering principle of Kalman (1960), for a linear measurement vector')
refs='''Amazon Science contributors. (n.d.). Chronos forecasting. Official implementation of normalization, Chronos-2 attention and quantile-loss computation. [Source repository](https://github.com/amazon-science/chronos-forecasting). Saved checkpoint parameters and the inspected implementation are distinguished from a complete historical software lock in Appendix A.

CatBoost developers. (n.d.). Regression objectives and common training parameters. [MAE objective](https://catboost.ai/docs/en/concepts/loss-functions-regression#MAE); [parameter definitions](https://catboost.ai/docs/en/references/training-parameters/common).

Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. Proceedings of the IEEE, 92(3), 401–422. [https://doi.org/10.1109/JPROC.2003.823141](https://doi.org/10.1109/JPROC.2003.823141)

Scikit-learn developers. (n.d.). Ridge. Official estimator documentation, objective and intercept convention. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

'''
s=s.replace('## Appendix A Reproducibility details',refs+'## Appendix A Reproducibility details')
# Sort the bibliography by author while preserving all URL text.
ra=s.index('## References')+len('## References');rb=s.index('## Appendix A')
entries=[p.strip() for p in s[ra:rb].split('\n\n') if p.strip()]
s=s[:ra]+'\n\n'+'\n\n'.join(sorted(entries,key=str.casefold))+'\n\n'+s[rb:]
s=s.replace('Historical software environment versions and hardware', 'The current inspected backbone implementation and saved checkpoint configuration specify the mathematical operations above, but they do not independently freeze every execution-time dependency. Historical software environment versions and hardware')
# Remove inline source-code formatting while retaining human-readable identifiers.
s=s.replace('`','')
s=s.replace('[https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html]', '[NeurIPS proceedings]')
(OUT/'NYX_Research_Manuscript.md').write_text(s,encoding='utf-8')
print('Revised manuscript',len(s.split()),'words and',counter[0],'numbered equations')
