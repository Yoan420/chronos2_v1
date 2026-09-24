# NYX for hourly day ahead electricity price forecasting across four European markets

Yoan Kesraoui

Correspondence: kesraoui.yoan@gmail.com

Research manuscript · Evaluation cutoff 31 August 2026

## Abstract

Pretrained time series models can incorporate market fundamentals, but their use in electricity price forecasting requires evaluation beyond average point accuracy. This study examines NYX, a hybrid pipeline combining a frozen Chronos-2 model, a rolling CatBoost residual correction and a governed state space correction. A retrospective prequential evaluation covers Belgium, Germany, France and the Netherlands from 16 September 2025 to 31 August 2026. On 33,596 jointly observed country–hour pairs, NYX achieves a mean absolute error of 11.189 EUR/MWh, compared with 12.197 for the Chronos-2 stage and 11.284 after CatBoost correction, an 8.27% reduction relative to the foundation model stage. A separate industrial reporting composite achieves 11.162 EUR/MWh, with benchmark vintage limitations preventing a strictly prospective comparison. The mathematical formulation distinguishes pretrained quantile inference, supervised nonlinear error adaptation and bounded online state correction. Their shared additive action preserves interval width, and nominal 80% intervals cover only 73.23–76.24% of observations across countries. Although the leading principal component explains 65.70% of simultaneous residual variance in an initial 180-day sample, a subsequent residual-learning experiment selects no additional correction on 95 validation days. None of the tested corrections improves pooled mean absolute error over NYX on the reserved 75-day segment. The evidence is conditional on reconstructed forecasts, available data snapshots and implementation limitations; it does not establish prospective trading value or state-of-the-art performance.

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

The target is the hourly day-ahead electricity price, in EUR/MWh, for Belgium (BE), Germany (DE), France (FR) and the Netherlands (NL). The analysis retains delivery dates from 16 September 2025 through 31 August 2026 inclusive, excluding every later observation. This gives 350 civil days, 8,400 physical hourly observations per country and 33,600 country–hour pairs in total. Delivery days contain 23, 24 or 25 hours as dictated by daylight-saving time; the sample contains one 23-hour day, one 25-hour day and 348 ordinary days. The cutoff limits the evaluated delivery dates; it does not assert that the retrospectively assembled forecast and observation snapshots were already available on that date.

Timestamps are aligned in UTC for scoring. Civil-hour diagnostics use the corresponding local European time zone. Both physical occurrences of the repeated autumn hour remain in the target sample. No target observation is interpolated. The external Storm reporting series lacks one observation per country at 00:00 UTC on 26 October 2025. Four-way comparisons between Chronos-2, the CatBoost-corrected stage, NYX and Storm therefore use exactly 8,399 hours per country, or 33,596 points. NYX interval diagnostics use all 33,600 points.

@table:data_description

The evaluation period crosses the transition of Single Day-Ahead Coupling to a 15-minute market time unit in October 2025 (NEMO Committee, 2025). EPEX SPOT documents a 60-minute index formed as the arithmetic average of the corresponding quarter-hour clearing prices (EPEX SPOT, 2025). The evaluated NYX configuration, however, reads an already hourly provider series. The retained evidence does not reconcile every post-transition value with the four native quarter-hour prices. Consequently, this paper evaluates the hourly price representation supplied by the source and does not establish either verified four-quarter aggregation or quarter-hour forecast accuracy.

### 2.2 Input variables and information timing

Each country's Chronos-2 task receives its own historical prices, calendar variables and six forecast covariates: residual load in France, Germany, Belgium, the Netherlands and Spain, and forecast nuclear generation in France. The physical covariates are expressed in GW. Spain is a covariate source, not an evaluated target market. The nuclear variable is forecast generation rather than realized output or an installed-capacity measure. The four price targets are evaluated separately, although they share the pretrained model and cross-country fundamental inputs.

@table:inputs

The intended forecast origin for delivery day D is 08:00 Paris time on D−1. The preparation code selects covariates using civil-time cutoffs and requires complete future covariate coverage. Historical day-ahead prices through the end of D−1 can in principle be known before their physical delivery, because they arise from an earlier auction. This economic timing does not imply that a subsequently downloaded historical price series is an exact reconstruction of its original publication vintage.

The retained artifacts are reconstructed prequential backtests. They are not an archive of 350 forecasts actually issued at their historical origins. Integrity checks validate archive identities, checksums, alignment and recorded temporal invariants, and report no target-day observations assimilated before their forecasts. These checks establish properties of the replay implementation. They do not establish original provider publication times: the source audits explicitly report that provider revision timestamps and production-grade point-in-time evidence are unavailable. Historical labels may also have been refreshed before the artifacts were frozen.

The frozen Chronos-2 checkpoint is an additional temporal limitation. Its paper was released in October 2025, after the start of the evaluation window (Ansari et al., 2025). The earlier portion of the replay must therefore be interpreted as retrospective evaluation of a later available model. The absence of pretraining overlap with the evaluated markets has not been established. This prevents interpreting the entire 350-day replay as a historically deployable zero-shot experiment.

Storm is retained as an industrial reporting comparator and is excluded from all NYX input features. Its archived reporting series combines a frozen day-ahead cache with a native curve recomputed where the cache is missing. The 350-day evaluation includes 456 such fallback hours in BE and DE and 480 in FR and NL. On the reserved 75-day segment, these counts are 72 and 96 respectively. These observations are retained in the main descriptive comparison and excluded in a sensitivity analysis. Even the cache-only subset does not independently prove availability at the strict historical 08:00 cutoff.

### 2.3 Mathematical formulation of the NYX pipeline

NYX denotes the nuclear-forecast pipeline with a frozen Chronos-2 stage, a supervised residual correction and a governed state space correction. The evaluated configuration contains no LoRA fine-tuning and no Storm blending. Figure 1 distinguishes the production cascade from the later residual experiment.

@figure:figure1_design|Figure 1. NYX stages and the separate residual-learning experiment through 31 August 2026. Both production-pipeline corrections shift all retained quantiles equally. The 180/95/75-day split applies only to the additional residual experiment.

#### 2.3.1 Forecasting task and frozen representation

The production corrections and governors are fitted separately for each country. We suppress the country index in their local equations; their states, noise estimates and daily weights remain country-specific. Cross-country dependence enters through the forecast covariates and the separately evaluated multivariate residual learners.

Let $c$ denote a country, $D$ a local delivery date and $h$ one of its physical delivery hours. Let $y_{cDh}$ be its price and $\mathcal I_D$ the information supplied at the declared origin, 08:00 on D−1. NYX first maps historical prices, calendar variables and six forecast fundamentals to quantiles:

$$
q^C_{cDh}(\tau)=F_{\theta,\tau}(\mathcal I_D),\qquad\theta=\theta_{\mathrm{pretrained}}.\tag{1}
$$

The superscript C identifies Chronos, while $\tau$ is a quantile level. The weights $\theta$ remain fixed. Information sharing inside this model is distinct from fitting downstream corrections. The supplied price context extends to D−1's final delivery hour; its admissibility relies on earlier day-ahead publication, not on delivery timestamps alone. Forecast-covariate vintages are reconstructed through as-of queries, whose timestamps do not independently certify provider publication.

Chronos-2 converts heterogeneous series to comparable numerical representations. For series $i$, observed historical positions $\mathcal O_i$ determine its location $\mu_i$ and population scale $s_i$:

$$
\begin{aligned}
\mu_i=\operatorname{mean}_{t\in\mathcal O_i}x_{it},\\
s_i^2=\operatorname{mean}_{t\in\mathcal O_i}(x_{it}-\mu_i)^2,\\
z_{it}=\operatorname{arsinh}((x_{it}-\mu_i)/s_i).
\end{aligned}\tag{2}
$$

An exactly zero scale is replaced by $10^{-5}$; completely missing histories receive location zero and scale one. Future-known values reuse historical moments. The monotone arcsinh transform accommodates negative prices while compressing large standardized magnitudes. This describes the verified backbone transformation, not a learned NYX clipping rule (Amazon Science contributors, n.d.).

The pinned checkpoint uses non-overlapping patches of $P=16$ steps. A learned residual multilayer perceptron embeds each patch:

$$
H^{(0)}_{ip}=f_{\mathrm{in}}([\mathbf j_p,\mathbf z_{ip},\mathbf m_{ip}])\in\mathbb R^{768}.\tag{3}
$$

Here $p$ indexes patches, $\mathbf j_p$ encodes positions, and $\mathbf m_{ip}$ distinguishes available values from zero-filled missing entries. Context and future patches are separated by a learned register token. NYX supplies 2,048 context hours, below the checkpoint's 8,192-step capacity. Its 12 layers alternate time attention, group attention and feed-forward transformations. Time attention links patches within a series; group attention links associated series at the same patch position (Ansari et al., 2025).

For one attention head, projected queries, keys and values are denoted by $Q,K,V$:

$$
\operatorname{Attn}(Q,K,V;M)=\operatorname{softmax}(QK^{\mathsf T}+M)V.\tag{4}
$$

The verified implementation sets attention scale to one. The additive mask $M$ blocks inadmissible keys; it is not a generic autoregressive triangular mask. Group identifiers allow a target to attend to its covariates. NYX's cross_learning=False prevents sharing across independent forecasting tasks, while preserving these within-task interactions. Time attention uses rotary positional encoding; group attention does not (Amazon Science contributors, n.d.).

An output head transforms future-patch representations into quantiles, then restores physical units:

$$
\begin{aligned}
\widehat Z_{ip}=f_{\mathrm{out}}(H^{(12)}_{ip})\in\mathbb R^{16|\mathcal Q|},\\
q^C_{it}(\tau)=\mu_i+s_i\sinh(\widehat z_{it}^{\tau}).
\end{aligned}\tag{5}
$$

The checkpoint's 21 levels are $\mathcal Q=\{0.01,0.99\}\cup\{0.05k:k=1,\ldots,19\}$. NYX requests nine deciles and retains 0.1, 0.5 and 0.9. Its backbone was trained with pinball errors on the transformed scale. With $u=z-\widehat z$, available-label mask $a$, known-future mask $k$, flattened training batch size $B$ and padded horizon $H'$, the verified reduction is:

$$
\begin{aligned}
\rho_\tau(u)=\tau\max(u,0)+(1-\tau)\max(-u,0),\\
\mathcal L=\frac{2}{BH'}\sum_{i,t}a_{it}(1-k_{it})
\sum_{\tau\in\mathcal Q}\rho_\tau(u_{it}^{\tau}).
\end{aligned}\tag{6}
$$

Known future covariates are excluded as prediction targets. This loss is neither a Gaussian likelihood nor an objective newly optimized by NYX. The mask multiplication precedes averaging over the full padded batch/horizon, rather than division by the number of unmasked entries (Amazon Science contributors, n.d.).

The architecture permits covariate-dependent forecasts without changing its weights: changing a supplied residual-load forecast changes attention inputs and hence the future representations. That is in-context conditioning, not a regression coefficient estimated anew for each country. A 16-hour patch is also a representation unit, not the market delivery resolution. A 23-, 24- or 25-hour day can be predicted through padded output patches and then restricted to its requested physical hours. Patch size therefore neither removes daylight-saving hours nor converts the response into a coarser price average.

#### 2.3.2 Supervised residual adaptation

For each country and delivery D, CatBoost fits past prediction errors on the hourly index set $\mathcal W_D=\{j:D-365\leq\operatorname{day}(j)<D\}$, with $n_D=|\mathcal W_D|$. Its target and empirical absolute-error objective are:

$$
\begin{aligned}
r_j=y_j-q^C_j(0.5),\\
\mathcal R_D(f)=\frac1{n_D}\sum_{j\in\mathcal W_D}|r_j-f(X_j)|.
\end{aligned}\tag{7}
$$

Index $j$ denotes a training hour; $X_j$ contains forecast fundamentals, Chronos outputs, calendar terms and forecast-profile interactions. The regressor therefore conditions on both the foundation forecast and its market context. It does not correct prices solely through a constant historical bias. Daily refitting uses at least 720 rows; otherwise the pipeline keeps Chronos unchanged.

An additive tree ensemble has the representation

$$
\begin{aligned}
f_M(X)=f_0(X)+\eta\sum_{m=1}^{M}T_m(X),\\
T_m(X)=\sum_{\ell}v_{m\ell}\mathbf1\{X\in R_{m\ell}\}.
\end{aligned}\tag{8}
$$

Each tree partitions feature space into leaves $R_{m\ell}$, with values $v_{m\ell}$. This representation explains nonlinear interactions: a leaf may distinguish a high foundation forecast jointly with an unusual residual-load profile. For absolute error, a negative subgradient away from zero is

$$
-\partial_f|r-f|=\operatorname{sign}(r-f).\tag{9}
$$

This gives the functional-gradient motivation for fitting successive errors; it is not a claim that CatBoost's complete split-search and leaf-estimation algorithm equals an elementary sign-fitting routine. MAE emphasizes the direction of remaining errors rather than their squared magnitude. (Prokhorenkova et al., 2018); (CatBoost developers, n.d.).

The executed settings are 700 iterations, depth 6, learning rate 0.03 and L2 leaf regularization 15. has_time=True preserves input order in relevant stages, but does not establish that boosting_type="Ordered" was selected. No such explicit option is forwarded. Likewise, a stored minimum-leaf setting is absent from the estimator constructor (CatBoost developers, n.d.).

The historical training pairs retain each day's upstream prediction, rather than replacing it with an in-sample forecast from a later CatBoost fit. This distinction matters: the model learns which errors accompanied the forecast that was actually reconstructed for that day. Profiles of the entire upcoming day are permissible only because they summarize forecast covariates and base predictions available under the declared origin protocol. They do not authorize using the upcoming day's realized load or prices. Apparent forecast availability remains conditional on the limitations of the historical source queries.

The fitted correction is bounded and shared across retained quantiles:

$$
\begin{aligned}
a_{cDh}=\operatorname{clip}(f_D(X_{cDh}),-40,40),\\
q^R_{cDh}(\tau)=q^C_{cDh}(\tau)+a_{cDh}.
\end{aligned}\tag{10}
$$

#### 2.3.3 Adaptive state correction and governance

The next layer tracks residual structure remaining after CatBoost. Its state space interpretation is

$$\xi_d=A\xi_{d-1}+\eta_d,\qquad z_{d,h}=\mathcal H(\xi_d,x_{d,h})+\varepsilon_{d,h}.\tag{11}$$

Here $\xi_d$ is a vector of latent correction coefficients, $\eta_d$ is zero-mean daily process noise with covariance $Q$, and $\varepsilon_{d,h}$ is zero-mean working observation noise with variance $R$. In the linear case, $\mathcal H(\xi,x)=x^{\mathsf T}\xi$; the unscented scale candidate uses the nonlinear map specified below. This is the filter's working model rather than an empirically verified generative law. Sequential conditioning updates the state moments after each training-hour label, while process noise is added at the daily transition.

 The market regressors are standardized using the preceding 365 days. For a training vector $v$, define

$$
\begin{aligned}
s(v)=\max(1.4826\operatorname{MAD}(v),\operatorname{SD}(v),10^{-6}),\\
\widetilde v=\operatorname{clip}((v-\operatorname{median}(v))/s(v),-5,5),\\
R=\max(s(y-q^R(0.5))^2,1).
\end{aligned}\tag{12}
$$

Here SD is the population standard deviation. Thus the scale includes a robust term but is not immune to extremes. $R$ is a shared working observation variance, not a demonstrated variance law for electricity-price errors.

Each candidate restarts for D, replays the 365-day window, and predicts every training day before assimilating its hourly labels. A daily state transition updates mean $m$ and covariance $P$:

$$
m^-=Am,\qquad P^-=APA^{\mathsf T}+Q.\tag{13}
$$

Linear candidates use $A=I$. Initial means are zero except the linear scale, initialized at one. Initial covariance is $RI$, with scale-coordinate variance $0.05^2$. Daily process variance is $0.001R$ per ordinary coordinate and $0.0001\times0.05^2$ for the scale coordinate. The nonlinear scale logit instead has daily persistence 0.99. These are fixed design settings, not EM estimates.

Table 10 specifies the five candidate maps: a scalar bias, a three-coefficient harmonic correction, a 14-coefficient market correction, and two bias-plus-scale corrections. Their state dimensions are therefore 1, 3, 14, 2 and 2. For scale candidates, the proposed correction relative to the CatBoost median is $b_0+(s-1)a_h$. The nonlinear version uses $s(u)=0.5+(1+e^{-u})^{-1}$; it remains between 0.5 and 1.5 for every finite logit $u$. This scale acts on the CatBoost shift, not on the electricity price itself.

The 14-dimensional market vector $x_h$ combines an intercept, two harmonics and eleven standardized descriptors: upstream median, interval width, CatBoost shift, five residual-load forecasts, their mean and range, and French nuclear generation. Linear scale is constrained to [0.5,1.5]. Bias/harmonic/market filters observe $y-q^R(0.5)$; scale filters observe $y-q^C(0.5)$, with measurement map $b_0+sa_h$.

Following the linear filtering principle of Kalman (1960), for a linear measurement vector $x$, scalar observation $z$ and prior state moments, the executed hourly update is

$$
\begin{aligned}
K=\frac{Px}{x^{\mathsf T}Px+R},\\
\widetilde\nu=\operatorname{clip}(z-x^{\mathsf T}m,-3\sqrt R,3\sqrt R),\\
m^+=m+K\widetilde\nu,\qquad P^+=P-Kx^{\mathsf T}P.
\end{aligned}\tag{14}
$$

Numerical covariance safeguards follow. Innovation clipping limits one observation's influence, but forfeits a literal claim of optimal Gaussian filtering. D's observations are never assimilated into D's forecast; no smoothing is used.

The gain is large when state uncertainty projected through the current regressor is large relative to observation noise. A large working $R$ reduces the influence of an individual residual; daily process noise prevents the model from becoming indefinitely certain about a fixed bias. These interpretations explain the update mechanism, not an empirically estimated stochastic law. In particular, clipped innovations, fixed noise ratios and daily restarts mean that standard optimality claims for a correctly specified linear Gaussian state model do not automatically apply to this overlay.

Unscented filtering approximates nonlinear transformations by propagating deterministic sigma points (Julier and Uhlmann, 2004). The archived nonlinear candidate propagates five sigma points through $\mathcal H(\xi)=b_0+s(u)a_h$, where $\xi=(b_0,u)^{\mathsf T}$, using centre weight 1/3 and other weights 1/6. If their transformed mean, observation variance including $R$, and state–observation cross-covariance are $\bar z,S,C$, its update is

$$
\begin{aligned}
z^*=\mathcal H(m)+\operatorname{clip}(z-\mathcal H(m),-3\sqrt R,3\sqrt R),\\
m^+=m+(C/S)(z^*-\bar z),\qquad P^+=P-CC^{\mathsf T}/S.
\end{aligned}\tag{15}
$$

The clipped innovation is centred at $\mathcal H(m)$, whereas the update uses $\bar z$. A newly identified sigma-point orientation discrepancy in the optimized implementation is documented in Appendix B; this archived candidate must not be described as an exact canonical UKF.

The governor chooses a candidate $k$ and weight $w$ using hourly MAE aggregated within the country over up to 60 replay days:

$$
\begin{aligned}
L_D(k,w)=\frac1{|\mathcal G_D|}\sum_{j\in\mathcal G_D}
|y_j-q^R_j(0.5)-w\operatorname{clip}(g_{k,j},-20,20)|,\\
w\in\{0,0.05,\ldots,1\}.
\end{aligned}\tag{16}
$$

Let $L_D^0$ be the identity loss. With at least 14 days, the minimizing pair is accepted only if

$$
\begin{aligned}
L_D^0-\min_{k,w}L_D(k,w)\geq\max(0.05,0.005L_D^0);\\
\text{otherwise }w_D=0.
\end{aligned}\tag{17}
$$

The same pair applies throughout D. There is no independent confirmation segment. Moreover, scaling and $R$ use the full 365-day window before its inner replay: the outer prediction excludes D, but inner governance is not fully nested validation.

#### 2.3.4 Interaction of the correction layers

The complete cascade is

$$
q^N_{cDh}(\tau)=q^C_{cDh}(\tau)+a_{cDh}+w_D\operatorname{clip}(g_{D,h},-20,20).\tag{18}
$$

Its components interact sequentially: Chronos outputs enter CatBoost, and the resulting median, width and shift enter Kalman. There is no joint loss, gradient flow back to Chronos, or LoRA update. The caps imply

$$
\begin{aligned}
|q^N(\tau)-q^C(\tau)|\leq60,\\
q^N(0.9)-q^N(0.1)=q^C(0.9)-q^C(0.1).
\end{aligned}\tag{19}
$$

Location can improve while dispersion remains inappropriate. These identities guarantee neither quantile calibration nor superior accuracy; lifting a saturated cap would require a new evaluation.

@table:configuration

### 2.4 Comparators and additional residual models

The main component comparison uses the three retained NYX stages on identical observations. These are sequential stage comparisons, not separately optimized model families, and they do not identify the causal importance of individual physical covariates. Storm is compared separately because its vintage history and forecasting mechanism differ. A seasonal-naïve predictor repeats the frozen price from the same civil hour seven delivery days earlier. Repeated lagged civil hours are averaged solely for this input feature; a missing lagged hour is excluded. After the first seven delivery days and remaining alignment exclusions, the five-model common-support comparison contains 8,230 hours per country across 343 delivery days.

The additional residual experiment asks whether remaining NYX errors are predictable from their own past, including cross-market information. These learners predict NYX minus the frozen observed price, and their output is subtracted from the NYX median. The validation search includes 13 candidates: the identity correction; country and country-by-hour exponentially weighted averages with half-lives of 7 or 28 days; univariate and multivariate ridge models with penalties 10 or 100; and principal-component ridge models with ranks one or two and the same penalties. This is a limited diagnostic family rather than an exhaustive architecture search.

The following equations specify the secondary residual learners and clarify why a contemporaneous factor structure need not yield a forecast gain. These learners act after the complete NYX cascade and are evaluated separately from its internal governor.

A separate diagnostic learner uses the opposite residual sign:

$$
\begin{aligned}
e_{cDh}=q^N_{cDh}(0.5)-y^{\mathrm{frozen}}_{cDh},\\
\widehat y^{\mathrm{extra}}_{cDh}=q^N_{cDh}(0.5)-\widehat e_{cDh}.
\end{aligned}\tag{20}
$$

Its features and fit labels are restricted through D−2. Features combine that day's same-hour residual with country and country/hour exponentially weighted means. For half-life $H$, a country summary obeys

$$
E_D=(1-\alpha)E_{D-1}+\alpha\bar e_{D-2},\qquad\alpha=1-2^{-1/H}.\tag{21}
$$

The hour-specific version updates the matching civil-hour summary. Ridge uses $H=14$, weekly refits, and up to 180 eligible days after warmup. After training-only imputation and normalization, its coefficients minimize

$$
\min_{\beta_0,B}\|E-\mathbf1\beta_0^{\mathsf T}-XB\|_F^2+\lambda\|B\|_F^2.\tag{22}
$$

The intercept is unpenalized. Univariate fits use three predictors per country; multivariate fits use twelve. This squared-error objective differs from the MAE used for model selection (Scikit-learn developers, n.d.).

This diagnostic deliberately separates learning representation from forecasting representation. Principal directions are estimated solely within the eligible fitting window; the realized target-day error cannot determine them. The same-hour feature uses the earlier day's civil-hour mean when that hour occurs twice, and an absent lagged hour is handled by training-only imputation for ridge. Daily features continue to update between weekly coefficient fits. Thus the coefficient vector can be temporarily fixed while its inputs change, and earlier evaluation labels can later enter a permitted rolling window. This is an adapting procedure, not a single forecast issued for the whole reserved segment.

PCA-ridge first standardizes the four training error series, with mean $\mu_e$ and diagonal scale matrix $S_e$, then extracts $r$ principal directions $V_r$:

$$
\begin{aligned}
Z=(E-\mathbf1\mu_e^{\mathsf T})S_e^{-1},\\
Z=U\Sigma V^{\mathsf T},\qquad T=ZV_r.
\end{aligned}\tag{23}
$$

Each of the three four-country predictor blocks undergoes the same target-based transformation and projection, producing $3r$ inputs. A second training-only standardization precedes ridge fitting. Predicted scores are reconstructed as

$$
\widehat E=\widehat T V_r^{\mathsf T}S_e+\mathbf1\mu_e^{\mathsf T}.\tag{24}
$$

PCA explains contemporaneous variance, not advance predictability. For any admissible information set, the covariance decomposition is

$$
\begin{aligned}
\operatorname{Cov}(e)=\operatorname{Cov}(\mathbb E[e\mid\mathcal I_D])\\
+\mathbb E[\operatorname{Cov}(e\mid\mathcal I_D)].
\end{aligned}\tag{25}
$$

Large principal components may belong mainly to the second, unpredictable term. Under squared loss, the conditional mean is optimal; under absolute loss, a conditional median is optimal. Consequently, strong simultaneous error correlation, accurate reconstruction or lower bias alone cannot establish an MAE improvement from another correction layer.

The first 180 days form an initial sample. The next 95 days, from 15 March to 17 June 2026, are used to select hyperparameters and the primary family by pooled MAE. Fitting and validation selection use frozen labels; the main reported evaluation scores use audited refreshed observations, with both label versions checked in sensitivity analysis. The final 75 retained days, from 18 June to 31 August 2026, are evaluated using the original saved choices. The cutoff revision neither reselects configurations nor retrains models. Ridge models use up to 180 eligible days and are refitted every seven days; fresh day-ahead features are constructed daily. A 14-day feature warm-up leaves 165 eligible days at the first validation refit. Imputation, normalization and principal components are estimated only from each admissible fitting window.

Earlier test-segment observations may subsequently enter fitting windows under the fixed D−2 rule. Thus, the final segment evaluates an adapting prequential procedure with fixed hyperparameter choices, not a single model frozen for 75 days. The selection record and code ordering establish selection before test scoring within this experiment; they do not establish external preregistration or non-use of this period in earlier NYX development. The retained test is a truncated subset of the previously examined test period, not a newly untouched holdout. All 145 fitting events contributing to the retained validation and test dates respect their label cutoff; the final test refit is on 27 August, using labels through 25 August. A targeted EWMA feature check is invariant to perturbations of forbidden recent and future labels; this check is not a proof covering every operation of the complete pipeline.

### 2.5 Error measures and statistical uncertainty

For an aligned sample of n pairs, the signed error is the forecast minus the observation. Mean absolute error is the primary point-forecast measure; root mean squared error emphasizes large errors, and mean error measures bias. All three retain the unit EUR/MWh. Percentage losses are avoided because the target can be zero or negative.

$$\operatorname{MAE}=\frac{1}{n}\sum_{i=1}^{n}|\widehat y_i-y_i|,\qquad \operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\widehat y_i-y_i)^2} \tag{26}$$

$$\operatorname{Bias}=\frac{1}{n}\sum_{i=1}^{n}(\widehat y_i-y_i). \tag{27}$$

Pooled scores are computed over the concatenated country–hour pairs. In the main full-support comparisons, country sample sizes coincide and pooled MAE is also the mean of country MAEs. Provenance sensitivities have unequal country counts and remain weighted by country–hour observations. Pooled RMSE is the square root of pooled squared error, rather than the arithmetic average of country RMSEs.

For lower and upper bounds l and u and median m, the central 80% interval score penalizes both width and observations outside the interval. The associated weighted interval score uses the median and this single interval (Bracher et al., 2021):

$$\operatorname{IS}_{0.2}=(u-l)+10\cdot\max(l-y,0)+10\cdot\max(y-u,0) \tag{28}$$

$$\operatorname{WIS}_{80}=\frac{0.5|y-m|+0.1\operatorname{IS}_{0.2}}{1.5}. \tag{29}$$

This single-interval WIS is not a full-distribution continuous ranked probability score. Lower values indicate better interval-and-median performance under the specified score. Empirical coverage and mean width are reported separately, since wide intervals can achieve coverage without being informative.

Negative-price diagnostics use realized prices below zero. High-price diagnostics use a country-specific 99th percentile estimated from frozen labels in the initial 180-day sample, then held fixed: 183.818 EUR/MWh in BE, 249.924 in DE, 158.461 in FR and 216.532 in NL. These are retrospective conditional diagnostics rather than ex ante event classifiers. In particular, high-price results for the full period include the sample used to estimate the threshold and are not an independent threshold-selection test. Exceedance rates need not remain at 1% as the price distribution changes.

Uncertainty is assessed with 2,000 paired moving-block bootstrap replicates using contiguous seven-day blocks, retaining the same resampled delivery days across all countries and compared models. Daily sums and counts reconstruct each pooled loss; both occurrences of daylight-saving hours remain included. Percentile 95% intervals follow the dependence-preserving principle of block resampling (Künsch, 1989). Full-period stage intervals are post hoc descriptive analyses. Reserved-segment intervals are conditional on the selected hyperparameters and that summer-dominated segment. Neither analysis incorporates uncertainty in data vintages, prior pipeline selection or unobserved future regimes. Secondary comparisons have no multiplicity adjustment.

## 3 Experimental results

### 3.1 Point forecasting performance through August 2026

NYX reduces MAE relative to its Chronos-2 stage in every country (Table 4). The pooled reduction is 1.008 EUR/MWh, from 12.197 to 11.189, or 8.27%. The paired 350-day seven-day block interval for NYX minus Chronos-2 is [−1.240, −0.715] EUR/MWh. Most of this improvement is already present after the CatBoost stage, whose pooled MAE is 11.284. The final governed filter reduces MAE by a further 0.094 EUR/MWh, with a descriptive interval [−0.176, −0.007]. These intervals describe the archived replay and do not correct for prior configuration selection.

@table:annual_stages

Relative to Storm's reporting composite, NYX reduces full-period MAE only in Germany and has higher full-period RMSE in all four countries. Its pooled MAE is 0.027 EUR/MWh higher, with a descriptive paired interval [−0.561, 0.736]. NYX's signed bias is closer to zero in all four countries. Thus, the evidence distinguishes improved centring from an improvement in the error distribution as a whole. The point estimates do not support a blanket assertion that NYX outperforms the industrial comparator.

@table:annual_benchmark

On the distinct weekly-naïve matched sample, pooled MAE is 33.645 EUR/MWh for the naïve predictor, 12.145 for Chronos-2, 11.278 for the supervised residual stage, 11.184 for NYX and 11.167 for Storm. This confirms that the pipeline improves substantially over a simple seasonal persistence reference on that support. It does not replace a comparison with strong separately fitted statistical and neural baselines. In particular, LEAR, independently tuned CatBoost price models and alternative foundation models have not been compared on this same four-country artifact set.

Figure 2 illustrates substantial temporal variation in performance. September 2025 is a partial month; the final bin covers all of August 2026. Each model is scored on exactly the same available hours within a bin. Such month-level differences are descriptive; the sample covers slightly less than a year and cannot establish persistent seasonal superiority.

@figure:figure2_monthly_mae|Figure 2. Monthly MAE on common observations from 16 September 2025 through 31 August 2026. September 2025 is partial. The external comparator contains recomputed fallback hours; its historical information set is not fully verified.

### 3.2 Extreme prices and interval reliability

The unconditional MAE hides a pronounced asymmetry. During high-price hours, NYX underpredicts on average in every country (Table 6). The German high-price subset has an MAE of 109.51 EUR/MWh and a bias of −95.02, despite an overall MAE near 11 EUR/MWh. The corresponding Dutch values are 95.79 and −74.47. France has many more exceedances of its fixed initial threshold but less severe average error, illustrating that exceedance counts and forecast difficulty are distinct quantities.

@table:extremes

The observed differences between countries should not be interpreted as estimates of a causal effect of generation mix or market coupling. These regime summaries condition on realized prices and use different country thresholds. They describe where the pipeline fails, while attribution to physical drivers would require additional controlled feature experiments or appropriately designed explanatory analyses.

Across all hours, nominal 80% intervals cover between 73.23% and 76.24% of observations (Table 7). Coverage during high-price hours falls to 29.41% in Germany and 43.81% in the Netherlands. Figure 3 locates more frequent interval misses among realized high-price observations. Because those subsets are selected using the realized target, neither their coverage nor their signed error establishes conditional miscalibration: even a calibrated predictive distribution can have both properties within an ex post upper-tail subset. These results motivate prospective event-risk evaluation, while undercoverage over the complete sample remains an observed marginal reliability limitation.

@table:intervals

@figure:figure3_interval_coverage|Figure 3. Empirical P10–P90 coverage over all hours and within realized negative- and high-price regimes. The dashed line is the nominal 80% coverage. High-price thresholds are fixed from the initial 180-day sample; conditional regime coverage is descriptive and need not equal unconditional nominal coverage even for a calibrated forecast.

The supervised correction reaches its absolute 40 EUR/MWh cap on only 0.12–0.52% of each country's hours, but on 16.47% of the German high-price hours and 11.43% of the Dutch high-price hours. The final governed correction does not reach its 20 EUR/MWh cap during the evaluated period. This pattern identifies cap saturation as a relevant diagnostic; it does not demonstrate that increasing the cap would improve held-out accuracy. More fundamentally, fixed-width location corrections cannot directly respond to a state-dependent increase in forecast dispersion.

### 3.3 Residual dependence and further correction

On the initial 180-day sample, the first principal component explains 65.70% of the variance of the four simultaneous NYX residual series and the first two explain 85.89%. The participation ratio is 2.07 within the four-dimensional space. After country standardization, the first component still explains 63.65%. Residual correlations are strongest between BE and NL (0.772) and DE and NL (0.710), whereas FR has weaker associations with DE (0.180) and NL (0.247). Figure 4 summarizes this dependence.

@figure:figure4_residual_dependence|Figure 4. Simultaneous residual dependence in the initial 180 days. Panel (a) shows country correlations. Panel (b) uses eigenvalues of the unstandardized four-country covariance matrix. These are retrospective descriptions of contemporaneous errors, not forecasts of latent factors or evidence of exploitable advance information.

Validation selects the identity correction, retaining NYX as the primary model. Its pooled validation MAE is 12.315 EUR/MWh; all tested additional families have higher validation MAE. Table 8 reports the best validation-selected configuration from each family on the reserved segment. These secondary configurations are not chosen by their final-segment performance.

On the reserved 75 days, none of the five validation-selected correction configurations reduces pooled MAE below NYX's 13.833 EUR/MWh. The country-level EWMA reduces the magnitude of signed bias, from −2.140 to −0.331, but raises pooled MAE to 13.861. The principal-component ridge model has an MAE increase of 0.237 EUR/MWh, with a paired interval [0.086, 0.427]. Its RMSE point estimate improves by only 0.036 EUR/MWh, with interval [−0.275, 0.359], which does not establish an RMSE gain. Multivariate ridge performs worse than univariate ridge by the primary point estimate. These pooled results do not imply that every individual-country point estimate is worse.

@table:residual_test

The result distinguishes simultaneous dependence from causal predictability. A low-dimensional representation can explain errors after their realization while failing to predict their signs and amplitudes at the next forecast origin. The experiment tests only the specified lagged-residual learners and losses. It does not exclude benefits from other admissible physical information, nonlinear methods or alternative adaptation strategies.

@figure:figure5_test_uncertainty|Figure 5. Reserved-segment MAE differences from NYX with paired 95% seven-day moving-block intervals. Positive values indicate larger loss than NYX. Secondary families are selected on validation; Storm is an external reporting comparison, not a candidate used to select the residual model. Intervals do not account for model-development or vintage uncertainty.

### 3.4 Sensitivity to observation and comparator provenance

Frozen and refreshed observations agree closely. After distinguishing float32 representation effects from substantive differences, there is one materially different hour per country, at 21:00 UTC on 16 June 2026. Absolute differences range from 6.955 to 9.305 EUR/MWh. The effect on full-period MAE is at most 0.00111 EUR/MWh in absolute value, and there is no material difference on the reserved segment. The cause of the discrepant versions is not identified here; numerical agreement cannot substitute for evidence of historical publication timing.

There are no alternative-source EPEX observation fallbacks within the retained dates. Table 9 therefore compares the complete matched sample with the subset excluding recomputed Storm values. The descriptive ranking of pooled MAE remains unchanged. This sensitivity reduces reliance on comparator fallbacks, while leaving the broader historical-vintage limitation unresolved.

@table:provenance_sensitivity

The cutoff-specific numerical audit recomputes all 70 reserved-segment score rows and 70 bootstrap rows from the saved forecasts, retaining both observation versions and the original validation-selected configurations. Stage comparisons use one shared support. No post-cutoff observations enter the revised statistics, resampled blocks or figures. Reproducibility of these calculations remains distinct from end-to-end prospective validation.

## 4 Discussion

### 4.1 What the hybrid structure contributes

The evidence supports a practical role for supervised residual learning around a frozen foundation model. Within the retained NYX pipeline, the CatBoost stage delivers most of the full-period MAE improvement, and the governed state space stage contributes a smaller additional reduction. The result is consistent with systematic errors remaining after covariate-informed foundation inference. It does not establish that this combination is preferable to training a market-specific model directly, because that independent baseline has not been fitted on the same sample and information set.

The additional residual experiment places a useful constraint on further complexity. Even after strong contemporaneous cross-country dependence is identified, the tested adaptive corrections fail to improve the selected primary loss. The zero-correction option is consequently a necessary reference. Its selection prevents a diagnostic observation about correlation from being converted into an unsupported claim about forecast improvement.

The supervised stage, filter and additional ridge experiments pursue different fitting criteria and operate at different layers. In particular, minimizing squared residual error in ridge fitting does not guarantee a lower evaluation MAE. The bias reduction achieved by EWMA also shows why a single signed-error measure is inadequate for selecting a model intended to predict hourly price levels.

### 4.2 Reliability and operational interpretation

Interval undercoverage is an observed limitation of the current system. Recentring the same base quantiles can reduce point error while retaining a dispersion model that is inappropriate for some market states. An operational probability statement should therefore depend on empirical recalibration and subsequent evaluation of interval scores, coverage and width. Increasing width mechanically would not, by itself, establish a better probabilistic forecast.

Large high-price errors and the greater weight they receive under RMSE explain why full-period MAE gains can coexist with weaknesses under other criteria. The appropriate operating choice depends on the decision problem and its costs. This study does not evaluate executable bids, positions, risk constraints, transaction costs or realized profit and loss. It therefore reports forecasting performance rather than economic value. The same distinction applies to the hourly aggregation: good hourly forecasts cannot establish skill for subhourly trading decisions.

The current diagnostic decomposition is transparent at the level of model stages, error regimes and correction magnitudes. It is not a SHAP analysis or a causal explanation of market drivers. The interpretability framework of Ma et al. (2026) could motivate a separate analysis, but its explanations and event-classification results cannot be transferred to NYX without running and validating the corresponding experiment.

### 4.3 Scope and threats to validity

The largest unresolved issue is the historical information set. Covariate cutoffs are enforced in the replay, yet provider revision timestamps are absent and production point-in-time proof is explicitly unavailable. The model checkpoint also postdates the beginning of the evaluated period. A prospective forecast archive or a replay using independently timestamped vintages is needed before interpreting the figures as attainable historical operational performance.

A second limitation is selection. The complete pipeline was developed before this manuscript and the 350-day sample is not established as an untouched holdout for that development. The extra 180/95/75-day experiment controls its own candidate selection, but cannot retroactively remove earlier choices from the NYX baseline. Full-period block intervals describe sampling variation conditional on the retained forecasts; they do not provide a selection-adjusted test of a new architecture.

Third, the markets are geographically and economically related, and the sample covers slightly less than a year. The reserved segment is dominated by summer conditions. Shared resampled blocks preserve cross-country dependence, but they do not create independent markets or additional years. The seven-day block length is a stated analysis choice rather than a demonstrated universal dependence horizon, and the numerous country and regime comparisons are exploratory.

Fourth, the comparison set is incomplete. The seasonal-naïve model is a basic reference and Storm has heterogeneous provenance. Strong independently trained baselines on a common data vintage, as advocated by Lago et al. (2021), remain necessary for a competitive benchmarking claim. Controlled input ablations would also be needed to quantify the contribution of nuclear and cross-country residual-load forecasts separately.

An implementation limitation also affects the nonlinear filter candidate: its sigma-point construction does not in general reproduce the intended covariance. Appendix B gives the mathematical discrepancy and its observed selection dates. The reported forecasts are unchanged, so these results concern the archived implementation; a corrected rerun could alter both candidate forecasts and governor decisions.

Finally, full public replication depends on access rights to the underlying market and provider data and the model artifacts. The local evidence package preserves aggregate results, analysis provenance and the scripts used for the manuscript; it does not establish that licensed data may be redistributed. These constraints limit the reproducibility claim to the available retained inputs and distinguish the present research manuscript from a completed public benchmark release.

## 5 Conclusion

This study evaluates an identifiable NYX pipeline for hourly day-ahead electricity prices in four European markets through 31 August 2026. In the retained 350-day replay, supervised residual adaptation and a governed filter reduce pooled MAE from 12.197 EUR/MWh for the Chronos-2 stage to 11.189 for NYX. Their mathematical interaction combines frozen quantile representation, nonlinear error learning and bounded state adaptation without joint training. The industrial reporting comparator remains competitive, and neither mean accuracy nor reduced bias eliminates large errors during price spikes. Equal shifts preserve the original quantile widths, and nominal 80% intervals under-cover across every country.

An additional validation-based experiment retains the unmodified NYX forecast despite substantial contemporaneous residual dependence. The tested lagged-residual corrections yield no further MAE improvement on the reserved segment. The empirical lesson is that cross-market error structure must be converted into admissible predictive information before it justifies an extra model layer. The next evidential requirements are historically verifiable input vintages, prospective evaluation, interval recalibration and stronger matched-information baselines. The current results support a carefully bounded case study of hybrid forecasting, rather than a claim of universal superiority or demonstrated trading profitability.

## Data and code availability

All reported numerical analyses use retained local NYX artifacts identified by the nuclear_forecast_v1 engine and civil_pit_v2 protocol, restricted to delivery dates no later than 31 August 2026. The source snapshots were assembled retrospectively after that cutoff; their original dates and checksums are retained in the evidence manifest. The accompanying package contains aggregate CSV tables, manuscript figures, source scripts and checksums. Its reproduction instructions identify the required local forecast matrix and archive inputs. Raw provider data and model weights are not publicly deposited with this manuscript, and redistribution permissions have not been established. Public access conditions and an archival repository identifier should be finalized before submission.

## Author declarations

The author list and institutional affiliations, funding statement, competing-interest declaration and permissions to publish provider-derived results require confirmation by the author before journal submission. This draft does not assert institutional endorsement, funding status or absence of competing interests. AI assistance was used to prepare text, analysis code and document formatting; responsibility for scientific verification and the final submitted manuscript remains with the author. No new NYX training run or prospective market experiment was performed in preparing this manuscript.

## References

Amazon Science contributors. (n.d.). Chronos forecasting. Official implementation of normalization, Chronos-2 attention and quantile-loss computation. [Source repository](https://github.com/amazon-science/chronos-forecasting). Saved checkpoint parameters and the inspected implementation are distinguished from a complete historical software lock in Appendix A.

Ansari, A. F., Shchur, O., Küken, J., et al. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821. [https://doi.org/10.48550/arXiv.2510.15821](https://doi.org/10.48550/arXiv.2510.15821)

Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2021). Evaluating epidemic forecasts in an interval format. PLOS Computational Biology, 17(2), e1008618. [https://doi.org/10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)

CatBoost developers. (n.d.). Regression objectives and common training parameters. [MAE objective](https://catboost.ai/docs/en/concepts/loss-functions-regression#MAE); [parameter definitions](https://catboost.ai/docs/en/references/training-parameters/common).

EPEX SPOT. (2025). 15-minute products in market coupling. Market design information and hourly price-index definition. [https://www.epexspot.com/en/new-15-minute-products-market-coupling](https://www.epexspot.com/en/new-15-minute-products-market-coupling)

Julier, S. J., & Uhlmann, J. K. (2004). Unscented filtering and nonlinear estimation. Proceedings of the IEEE, 92(3), 401–422. [https://doi.org/10.1109/JPROC.2003.823141](https://doi.org/10.1109/JPROC.2003.823141)

Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1), 35–45. [https://doi.org/10.1115/1.3662552](https://doi.org/10.1115/1.3662552)

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. The Annals of Statistics, 17(3), 1217–1241. [https://doi.org/10.1214/aos/1176347265](https://doi.org/10.1214/aos/1176347265)

Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy, 293, 116983. [https://doi.org/10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)

Ma, J., Chen, Y.-w., & Meng, F. (2026). An interpretable machine learning approach for forecasting the occurrence of extreme electricity prices in the day-ahead market. Journal of the Operational Research Society. Advance online publication. [https://doi.org/10.1080/01605682.2026.2660989](https://doi.org/10.1080/01605682.2026.2660989)

NEMO Committee. (2025). Single Day-Ahead Coupling: implementation of the 15-minute market time unit. Official communications dated 1 and 7 October 2025, listed on the SDAC resource page. [https://www.nemo-committee.eu/sdac](https://www.nemo-committee.eu/sdac)

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)

Scikit-learn developers. (n.d.). Ridge. Official estimator documentation, objective and intercept convention. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## Appendix A Reproducibility details

The retained foundation checkpoint is amazon/chronos-2 at revision 29ec3766d36d6f73f0696f85560a422f50e8498c. The evaluated output median is the final residual_kalman quantile 0.5; the diagnostic stages are the corresponding Chronos-2 and residual_corrected medians from the same artifact. Identical names from another run or configuration are not interchangeable with these outputs.

The source artifact includes a preceding warm-up and daily rolling refits. The first 180 evaluation days used by the additional residual experiment are therefore not the foundation model's pretraining sample or the only fitting history used by the complete NYX pipeline. This distinction is essential when reproducing the analysis.

The bootstrap uses daily country-specific sums of absolute and squared error and sample counts, draws shared contiguous seven-day blocks and truncates the concatenated draws to the required number of days. All revised bootstrap calculations use 2,000 replicates and the fixed original seed 20260915; the integer is a reproducibility setting, not the evaluation cutoff. No hour is treated as independent merely to narrow an uncertainty interval.

All table figures are rounded only for display. Supporting CSV files retain full numerical precision. Positive bias means overprediction; in the reserved-segment table and forest plot, a positive MAE difference means worse performance than NYX. Full-period stage comparisons explicitly report NYX minus the stated reference. The same distinction applies when reading the corresponding bootstrap tables.

Model configuration values and feature details are reconstructed from retained configuration files and the code audit. A recorded minimum-leaf parameter is not listed as an effective CatBoost setting because it is not forwarded by the actual estimator constructor. The current inspected backbone implementation and saved checkpoint configuration specify the mathematical operations above, but they do not independently freeze every execution-time dependency. Historical software environment versions and hardware for full NYX training have not been comprehensively archived in this manuscript. Timings from the lightweight residual audit are therefore not presented as end-to-end NYX computational cost.

## Appendix B State space candidates and numerical limitation

The upstream median is the CatBoost-corrected median and $a$ denotes the CatBoost shift. Each candidate state is reinitialized for delivery D and replayed over its admissible 365-day history. Table 10 specifies the correction relative to that upstream median; their predicted observation is $b_0+sa$ and their assimilated target is the Chronos-relative error $y-q^C(0.5)$.

@table:filters

The optimized nonlinear candidate has a sigma-point orientation discrepancy. Let $L$ be the lower-triangular NumPy Cholesky factor, so that $P=LL^{\mathsf T}$. The code builds its symmetric offsets from rows of $L$, whereas covariance-preserving sigma points require its columns. With the implemented weights, the represented covariance is $L^{\mathsf T}L$, which generally differs from $P$. For example, $P=\left[\begin{matrix}4&1\\1&1\end{matrix}\right]$ gives represented covariance approximately $\left[\begin{matrix}4.25&0.433\\0.433&0.75\end{matrix}\right]$. This is a read-only analytic check, not a replacement forecast experiment.

Within the retained dates, this candidate is selected only for the Netherlands on 21 February 2026, comprising 24 hours at weight 0.45. It is never selected in the reserved 75-day segment. The small observed selection count does not bound the effect of a corrected implementation: altered candidate losses could change governance choices on other dates. The manuscript therefore reports the archived governed implementation, not a validated canonical UKF. No forecast has been silently repaired. A corrected replay and renewed evaluation are required before an accuracy claim is attributed to the standard unscented filter.
