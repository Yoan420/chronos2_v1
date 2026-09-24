# Mechanisms, interactions and limits of the NYX cascade

Editorial scope: theory for the manuscript censored at 31 August 2026. Dates after that cutoff must not enter its reported evaluation. This section describes the retained implementation; it does not imply a new training run, a new checkpoint or historically certified data availability. The 24 equation blocks use identifiers T1–T25, with T13 replaced by the candidate table; renumber during integration.

#### 2.3.1 Forecasting task and frozen representation

Let $c$ denote a country, $D$ a local delivery date and $h$ one of its physical delivery hours. Let $y_{cDh}$ be its price and $\mathcal I_D$ the information supplied at the declared origin, 08:00 on D−1. NYX first maps historical prices, calendar variables and six forecast fundamentals to quantiles:

$$
q^C_{cDh}(\tau)=F_{\theta,\tau}(\mathcal I_D),\qquad\theta=\theta_{\mathrm{pretrained}}.\tag{T1}
$$

The superscript C identifies Chronos, while $\tau$ is a quantile level. The weights $\theta$ remain fixed. Each country has separately fitted residual and filter stages; suppressed country indices, including on governance weights, remain country-specific. Information sharing inside this model is distinct from fitting downstream corrections. The supplied price context extends to D−1's final delivery hour; its admissibility relies on earlier day-ahead publication, not on delivery timestamps alone. Forecast-covariate vintages are reconstructed through as-of queries, whose timestamps do not independently certify provider publication.

Chronos-2 converts heterogeneous series to comparable numerical representations. For series $i$, observed historical positions $\mathcal O_i$ determine its location $\mu_i$ and population scale $s_i$:

$$
\begin{aligned}
\mu_i=\operatorname{mean}_{t\in\mathcal O_i}x_{it},\\
s_i^2=\operatorname{mean}_{t\in\mathcal O_i}(x_{it}-\mu_i)^2,\\
z_{it}=\operatorname{arsinh}((x_{it}-\mu_i)/s_i).
\end{aligned}\tag{T2}
$$

An exactly zero scale is replaced by $10^{-5}$; completely missing histories receive location zero and scale one. Future-known values reuse historical moments. The monotone arcsinh transform accommodates negative prices while compressing large standardized magnitudes. This describes the verified backbone transformation, not a learned NYX clipping rule. [Official normalization implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py#L86-L126).

The pinned checkpoint uses non-overlapping patches of $P=16$ steps. A learned residual multilayer perceptron embeds each patch:

$$
H^{(0)}_{ip}=f_{\mathrm{in}}([\mathbf j_p,\mathbf z_{ip},\mathbf m_{ip}])\in\mathbb R^{768}.\tag{T3}
$$

Here $p$ indexes patches, $\mathbf j_p$ encodes positions, and $\mathbf m_{ip}$ distinguishes available values from zero-filled missing entries. Context and future patches are separated by a learned register token. NYX supplies 2,048 context hours, below the checkpoint's 8,192-step capacity. Its 12 layers alternate time attention, group attention and feed-forward transformations. Time attention links patches within a series; group attention links associated series at the same patch position. [Ansari et al., 2025, Sections 3.1–3.2](https://arxiv.org/html/2510.15821v1#S3).

For one attention head, projected queries, keys and values are denoted by $Q,K,V$:

$$
\operatorname{Attn}(Q,K,V;M)=\operatorname{softmax}(QK^{\mathsf T}+M)V.\tag{T4}
$$

The verified implementation sets attention scale to one. The additive mask $M$ blocks inadmissible keys; it is not a generic autoregressive triangular mask. Group identifiers allow a target to attend to its covariates. NYX's `cross_learning=False` prevents sharing across independent forecasting tasks, while preserving these within-task interactions. Time attention uses rotary positional encoding; group attention does not. [Official attention layers](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/layers.py#L217-L370).

An output head transforms future-patch representations into quantiles, then restores physical units:

$$
\begin{aligned}
\widehat Z_{ip}=f_{\mathrm{out}}(H^{(12)}_{ip})\in\mathbb R^{16|\mathcal Q|},\\
q^C_{it}(\tau)=\mu_i+s_i\sinh(\widehat z_{it}^{\tau}).
\end{aligned}\tag{T5}
$$

The checkpoint's 21 levels are $\mathcal Q=\{0.01,0.99\}\cup\{0.05k:k=1,\ldots,19\}$. NYX requests nine deciles and retains 0.1, 0.5 and 0.9. Its backbone was trained with pinball errors on the transformed scale. With $u=z-\widehat z$, available-label mask $a$, known-future mask $k$, flattened training batch size $B$ and padded horizon $H'$, the verified reduction is:

$$
\begin{aligned}
\rho_\tau(u)=\tau\max(u,0)+(1-\tau)\max(-u,0),\\
\mathcal L=\frac{2}{BH'}\sum_{i,t}a_{it}(1-k_{it})
\sum_{\tau\in\mathcal Q}\rho_\tau(u_{it}^{\tau}).
\end{aligned}\tag{T6}
$$

Known future covariates are excluded as prediction targets. This loss is neither a Gaussian likelihood nor an objective newly optimized by NYX. The mask multiplication precedes averaging over the full padded batch/horizon, rather than division by the number of unmasked entries. [Official output and loss implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py#L473-L518).

The architecture permits covariate-dependent forecasts without changing its weights: changing a supplied residual-load forecast changes attention inputs and hence the future representations. That is in-context conditioning, not a regression coefficient estimated anew for each country. A 16-hour patch is also a representation unit, not the market delivery resolution. A 23-, 24- or 25-hour day can be predicted through padded output patches and then restricted to its requested physical hours. Patch size therefore neither removes daylight-saving hours nor converts the response into a coarser price average.

#### 2.3.2 Supervised residual adaptation

For each D, CatBoost fits past prediction errors using dates $\mathcal W_D=\{D-365,\ldots,D-1\}$. Its target and empirical absolute-error objective are:

$$
\begin{aligned}
r_j=y_j-q^C_j(0.5),\\
\mathcal R_D(f)=\frac1{n_D}\sum_{j\in\mathcal W_D}|r_j-f(X_j)|.
\end{aligned}\tag{T7}
$$

Index $j$ denotes a training hour; $X_j$ contains forecast fundamentals, Chronos outputs, calendar terms and forecast-profile interactions. The regressor therefore conditions on both the foundation forecast and its market context. It does not correct prices solely through a constant historical bias. Daily refitting uses at least 720 rows; otherwise the pipeline keeps Chronos unchanged.

An additive tree ensemble has the representation

$$
\begin{aligned}
f_M(X)=f_0(X)+\eta\sum_{m=1}^{M}T_m(X),\\
T_m(X)=\sum_{\ell}v_{m\ell}\mathbf1\{X\in R_{m\ell}\}.
\end{aligned}\tag{T8}
$$

Each tree partitions feature space into leaves $R_{m\ell}$, with values $v_{m\ell}$. This representation explains nonlinear interactions: a leaf may distinguish a high foundation forecast jointly with an unusual residual-load profile. For absolute error, a negative subgradient away from zero is

$$
-\partial_f|r-f|=\operatorname{sign}(r-f).\tag{T9}
$$

This gives the functional-gradient motivation for fitting successive errors; it is not a claim that CatBoost's complete split-search and leaf-estimation algorithm equals an elementary sign-fitting routine. MAE emphasizes the direction of remaining errors rather than their squared magnitude. [CatBoost paper](https://arxiv.org/abs/1706.09516); [official MAE definition](https://catboost.ai/docs/en/concepts/loss-functions-regression#MAE).

The executed settings are 700 iterations, depth 6, learning rate 0.03 and L2 leaf regularization 15. `has_time=True` preserves input order in relevant stages, but does not establish that `boosting_type="Ordered"` was selected. No such explicit option is forwarded. Likewise, a stored minimum-leaf setting is absent from the estimator constructor. [Official parameter distinctions](https://catboost.ai/docs/en/references/training-parameters/common#has_time).

The historical training pairs retain each day's upstream prediction, rather than replacing it with an in-sample forecast from a later CatBoost fit. This distinction matters: the model learns which errors accompanied the forecast that was actually reconstructed for that day. Profiles of the entire upcoming day are permissible only because they summarize forecast covariates and base predictions available under the declared origin protocol. They do not authorize using the upcoming day's realized load or prices. Apparent forecast availability remains conditional on the limitations of the historical source queries.

The fitted correction is bounded and shared across retained quantiles:

$$
\begin{aligned}
a_{cDh}=\operatorname{clip}(f_D(X_{cDh}),-40,40),\\
q^R_{cDh}(\tau)=q^C_{cDh}(\tau)+a_{cDh}.
\end{aligned}\tag{T10}
$$

#### 2.3.3 Adaptive state correction and governance

The next layer tracks residual structure remaining after CatBoost. Its market regressors are standardized using the preceding 365 days. For a training vector $v$, define

$$
\begin{aligned}
s(v)=\max(1.4826\operatorname{MAD}(v),\operatorname{SD}(v),10^{-6}),\\
\widetilde v=\operatorname{clip}((v-\operatorname{median}(v))/s(v),-5,5),\\
R=\max(s(y-q^R(0.5))^2,1).
\end{aligned}\tag{T11}
$$

Here SD is the population standard deviation. Thus the scale includes a robust term but is not immune to extremes. $R$ is a shared working observation variance, not a demonstrated variance law for electricity-price errors.

Each candidate restarts for D, replays the 365-day window, and predicts every training day before assimilating its hourly labels. A daily state transition updates mean $m$ and covariance $P$:

$$
m^-=Am,\qquad P^-=APA^{\mathsf T}+Q.\tag{T12}
$$

Linear candidates use $A=I$. Initial means are zero except the linear scale, initialized at one. Initial covariance is $RI$, with scale-coordinate variance $0.05^2$. Daily process variance is $0.001R$ per ordinary coordinate and $0.0001\times0.05^2$ for the scale coordinate. The nonlinear scale logit instead has daily persistence 0.99. These are fixed design settings, not EM estimates.

Table 10 specifies the five candidate maps: a scalar bias, a three-coefficient harmonic correction, a 14-coefficient market correction, and two bias-plus-scale corrections. Their state dimensions are therefore 1, 3, 14, 2 and 2. For scale candidates, the proposed correction relative to the CatBoost median is $b_0+(s-1)a_h$. The nonlinear version uses $s(u)=0.5+(1+e^{-u})^{-1}$; it remains between 0.5 and 1.5 for every finite logit $u$. This scale acts on the CatBoost shift, not on the electricity price itself.

The 14-dimensional market vector $x_h$ combines an intercept, two harmonics and eleven standardized descriptors: upstream median, interval width, CatBoost shift, five residual-load forecasts, their mean and range, and French nuclear generation. Linear scale is constrained to [0.5,1.5]. Bias/harmonic/market filters observe $y-q^R(0.5)$; scale filters observe $y-q^C(0.5)$, with measurement map $b_0+sa_h$.

For a linear measurement vector $x$, scalar observation $z$ and prior state moments, the executed hourly update is

$$
\begin{aligned}
K=\frac{Px}{x^{\mathsf T}Px+R},\\
\widetilde\nu=\operatorname{clip}(z-x^{\mathsf T}m,-3\sqrt R,3\sqrt R),\\
m^+=m+K\widetilde\nu,\qquad P^+=P-Kx^{\mathsf T}P.
\end{aligned}\tag{T14}
$$

Numerical covariance safeguards follow. Innovation clipping limits one observation's influence, but forfeits a literal claim of optimal Gaussian filtering. D's observations are never assimilated into D's forecast; no smoothing is used.

The gain is large when state uncertainty projected through the current regressor is large relative to observation noise. A large working $R$ reduces the influence of an individual residual; daily process noise prevents the model from becoming indefinitely certain about a fixed bias. These interpretations explain the update mechanism, not an empirically estimated stochastic law. In particular, clipped innovations, fixed noise ratios and daily restarts mean that standard optimality claims for a correctly specified linear Gaussian state model do not automatically apply to this overlay.

The nonlinear candidate propagates five sigma points through $h(\theta)=b_0+s(u)a_h$, using centre weight 1/3 and other weights 1/6. If their transformed mean, observation variance including $R$, and state–observation cross-covariance are $\bar z,S,C$, its update is

$$
\begin{aligned}
z^*=h(m)+\operatorname{clip}(z-h(m),-3\sqrt R,3\sqrt R),\\
m^+=m+(C/S)(z^*-\bar z),\qquad P^+=P-CC^{\mathsf T}/S.
\end{aligned}\tag{T15}
$$

The clipped innovation is centred at $h(m)$, whereas the update uses $\bar z$. A newly identified sigma-point orientation discrepancy in the optimized implementation is documented below; this archived candidate must not be described as an exact canonical UKF.

The governor chooses a candidate $k$ and weight $w$ using pooled hourly MAE over up to 60 replay days:

$$
\begin{aligned}
L_D(k,w)=\frac1{|\mathcal G_D|}\sum_{j\in\mathcal G_D}
|y_j-q^R_j(0.5)-w\operatorname{clip}(g_{k,j},-20,20)|,\\
w\in\{0,0.05,\ldots,1\}.
\end{aligned}\tag{T16}
$$

Let $L_D^0$ be the identity loss. With at least 14 days, the minimizing pair is accepted only if

$$
\begin{aligned}
L_D^0-\min_{k,w}L_D(k,w)\geq\max(0.05,0.005L_D^0);\\
\text{otherwise }w_D=0.
\end{aligned}\tag{T17}
$$

The same pair applies throughout D. There is no independent confirmation segment. Moreover, scaling and $R$ use the full 365-day window before its inner replay: the outer prediction excludes D, but inner governance is not fully nested validation.

#### 2.3.4 Interaction of the correction layers

The complete cascade is

$$
q^N_{cDh}(\tau)=q^C_{cDh}(\tau)+a_{cDh}+w_D\operatorname{clip}(g_{D,h},-20,20).\tag{T18}
$$

Its components interact sequentially: Chronos outputs enter CatBoost, and the resulting median, width and shift enter Kalman. There is no joint loss, gradient flow back to Chronos, or LoRA update. The caps imply

$$
\begin{aligned}
|q^N(\tau)-q^C(\tau)|\leq60,\\
q^N(0.9)-q^N(0.1)=q^C(0.9)-q^C(0.1).
\end{aligned}\tag{T19}
$$

Location can improve while dispersion remains inappropriate. These identities guarantee neither quantile calibration nor superior accuracy; lifting a saturated cap would require a new evaluation.

### 2.4 Predictability of the remaining errors

A separate diagnostic learner uses the opposite residual sign:

$$
\begin{aligned}
e_{cDh}=q^N_{cDh}(0.5)-y^{\mathrm{frozen}}_{cDh},\\
\widehat y^{\mathrm{extra}}_{cDh}=q^N_{cDh}(0.5)-\widehat e_{cDh}.
\end{aligned}\tag{T20}
$$

Its features and fit labels are restricted through D−2. Features combine that day's same-hour residual with country and country/hour exponentially weighted means. For half-life $H$, a country summary obeys

$$
E_D=(1-\alpha)E_{D-1}+\alpha\bar e_{D-2},\qquad\alpha=1-2^{-1/H}.\tag{T21}
$$

The hour-specific version updates the matching civil-hour summary. Ridge uses $H=14$, weekly refits, and up to 180 eligible days after warmup. After training-only imputation and normalization, its coefficients minimize

$$
\min_{\beta_0,B}\|E-\mathbf1\beta_0^{\mathsf T}-XB\|_F^2+\lambda\|B\|_F^2.\tag{T22}
$$

The intercept is unpenalized. Univariate fits use three predictors per country; multivariate fits use twelve. This squared-error objective differs from the MAE used for model selection. [Official Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

This diagnostic deliberately separates learning representation from forecasting representation. Principal directions are estimated solely within the eligible fitting window; the realized target-day error cannot determine them. The same-hour feature uses the earlier day's civil-hour mean when that hour occurs twice, and an absent lagged hour is handled by training-only imputation for ridge. Daily features continue to update between weekly coefficient fits. Thus the coefficient vector can be temporarily fixed while its inputs change, and earlier evaluation labels can later enter a permitted rolling window. This is an adapting procedure, not a single forecast issued for the whole reserved segment.

PCA-ridge first standardizes the four training error series, with mean $\mu_e$ and diagonal scale matrix $S_e$, then extracts $r$ principal directions $V_r$:

$$
\begin{aligned}
Z=(E-\mathbf1\mu_e^{\mathsf T})S_e^{-1},\\
Z=U\Sigma V^{\mathsf T},\qquad T=ZV_r.
\end{aligned}\tag{T23}
$$

Each of the three four-country predictor blocks undergoes the same target-based transformation and projection, producing $3r$ inputs. A second training-only standardization precedes ridge fitting. Predicted scores are reconstructed as

$$
\widehat E=\widehat T V_r^{\mathsf T}S_e+\mathbf1\mu_e^{\mathsf T}.\tag{T24}
$$

PCA explains contemporaneous variance, not advance predictability. For any admissible information set, the covariance decomposition is

$$
\begin{aligned}
\operatorname{Cov}(e)=\operatorname{Cov}(\mathbb E[e\mid\mathcal I_D])\\
+\mathbb E[\operatorname{Cov}(e\mid\mathcal I_D)].
\end{aligned}\tag{T25}
$$

Large principal components may belong mainly to the second, unpredictable term. Under squared loss, the conditional mean is optimal; under absolute loss, a conditional median is optimal. Consequently, strong simultaneous error correlation, accurate reconstruction or lower bias alone cannot establish an MAE improvement from another correction layer.

# Evidence notes and integration cautions

These notes accompany the theoretical section and are not new experimental results.

1. **New numerical limitation requiring disclosure.** In `chronos2_hourly/kalman_residual.py:1239–1242`, let $L=\operatorname{chol}_{\mathrm{NumPy}}(P)$, a lower-triangular factor. The optimized code adds rows of $L$, whereas moment-preserving sigma points require its columns. Their weighted covariance is therefore $L^{\mathsf T}L$, generally different from $LL^{\mathsf T}=P$. A read-only analytic check with `P=[[4,1],[1,1]]` produces `[[4.25,0.4330127],[0.4330127,0.75]]`. The installed `pykalman/unscented.py:62–116` uses SciPy's upper-triangular Cholesky and a different array orientation. No model or archived forecast was changed. The retained outputs select this candidate for NL on 21 February and 3 September 2026; only the first date, 24 NL hours, remains before the 31 August cutoff. This identifies directly selected hours, not a bound on the impact of a hypothetical corrected rerun. The unscented candidate competes in governance on other dates too. Do not silently correct its description while retaining old results, or claim empirical equivalence to a standard UKF.

2. **Exact backbone versus publication description.** The normalization, mask reduction, attention scale and patch-concatenation order above were cross-checked against official source and the installed Chronos implementation, not inferred from generic Transformer equations. The checkpoint's configuration confirms patch/stride 16, arcsinh, 12 layers and 21 quantiles. Current installed code and a checkpoint hash do not independently freeze all execution-time library versions. The factor two/full-padded-horizon reduction in T6 is the implementation expression; the paper presents the underlying pinball objective without that exact reduction detail.

3. **Source and publication chronology.** The checkpoint is `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`; its paper was first submitted 17 October 2025. Censoring evaluation at 31 August 2026 does not establish historical deployability at early origins, eliminate uncertain price/covariate vintages, or certify pretraining exclusion.

4. **Local evidence map.** Backbone invocation: `chronos2_modular/forecasting.py:397–410`, quantile request `chronos2_modular/common.py:20`. Residual constructor/fit: `chronos2_hourly/models/residual_corrector.py:1029–1046,1134–1145,1177–1204`; rolling dates `chronos2_hourly/nuclear_forecast.py:272–387`. State and candidate equations: `chronos2_hourly/kalman_residual.py:793–1010,1132–1263`; governance and full-window preprocessing: `1287–1396,1474–1624`. Additional error sign, features, ridge and PCA: `tmp/tensor_timesfm_audit/metrics/experiments.py:48–66,72–86,102–151,197–237`. The latter is the original algorithmic source; revised cutoff-specific scoring must be cited to the new analysis artifacts. Training labels for the additional experiment are frozen labels; validation uses those labels, while reported scoring-label versions must be declared separately.

# Primary references and URLs

- Ansari, A. F., Shchur, O., Küken, J., et al. (2025). *Chronos-2: From Univariate to Universal Forecasting*. [Paper](https://arxiv.org/abs/2510.15821), [architecture and objective](https://arxiv.org/html/2510.15821v1#S3), [official repository](https://github.com/amazon-science/chronos-forecasting), [group construction](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/dataset.py#L224-L240), [cross-learning switch](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py#L583-L600).
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. Advances in Neural Information Processing Systems, 31. [Authors' paper](https://arxiv.org/abs/1706.09516). This citation describes CatBoost; ordered boosting must not be attributed to the evaluated run without its effective parameter evidence.
- CatBoost developers. *Regression objectives* and *Common training parameters*. [MAE](https://catboost.ai/docs/en/concepts/loss-functions-regression), [has_time and boosting_type](https://catboost.ai/docs/en/references/training-parameters/common).
- pykalman developers. *Kalman and Unscented Kalman Filter documentation*. [Official documentation](https://pykalman.readthedocs.io/en/latest/index.html). The exact optimized implementation was checked locally against installed pykalman 0.11.2, not assumed identical to this documentation's displayed version.
- Scikit-learn developers. *Ridge*. [Official objective and intercept convention](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). The local prototype explicitly uses the SVD solver; the linked current documentation is conceptual support, not an execution-time version record.
