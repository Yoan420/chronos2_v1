import numpy as np
import tensorly as tl
import os
import matplotlib.pyplot as plt


def _make_stable_var1_matrix(
    rank_dimension: int,
    ar_strength: float,
    random_offdiag_scale: float = 0.08,
    stability_radius: float = 0.98,
) -> np.ndarray:
    """
    Build a stable-ish VAR(1) matrix Phi in R^{rank x rank}.

    Shapes:
        rank_dimension = R
        return Phi: (R, R)

    Meaning:
        c_t = Phi @ c_{t-1} + s_t + eps_t
        Larger ar_strength -> stronger time dependence.

    Notes:
        We create a random matrix, make diagonal strong, then rescale by spectral radius
        so that Phi is (roughly) stable (eigenvalues magnitude < 1).
    """
    if ar_strength <= 0.0:
        return np.zeros((rank_dimension, rank_dimension), dtype=float)

    var1_base_matrix = np.random.uniform(
        low=-random_offdiag_scale,
        high=random_offdiag_scale,
        size=(rank_dimension, rank_dimension),
    ).astype(float)

    diagonal_values = np.random.uniform(0.6, 0.9, size=(rank_dimension,))
    np.fill_diagonal(var1_base_matrix, diagonal_values)

    eigenvalues = np.linalg.eigvals(var1_base_matrix)
    spectral_radius = np.max(np.abs(eigenvalues))

    if spectral_radius > 1e-12:
        var1_stable_matrix = (stability_radius / spectral_radius) * var1_base_matrix
    else:
        var1_stable_matrix = var1_base_matrix

    var1_transition_matrix = ar_strength * var1_stable_matrix
    return var1_transition_matrix


def _make_equicorr_cov(
    rank_dimension: int,
    noise_std: float,
    rho: float,
) -> np.ndarray:
    """
    Equicorrelation covariance matrix:

        Sigma = noise_std^2 * ( (1-rho) I + rho * 11^T )

    Shapes:
        Sigma: (R, R)

    Meaning:
        Correlated Gaussian noise across the R latent time dimensions.

    Constraint:
        rho must be in (-1/(R-1), 1) to keep Sigma positive definite.
    """
    rho_lower_bound = -1.0 / (rank_dimension - 1) + 1e-6
    rho = float(np.clip(rho, rho_lower_bound, 0.99))

    identity_matrix = np.eye(rank_dimension, dtype=float)
    ones_matrix = np.ones((rank_dimension, rank_dimension), dtype=float)

    noise_covariance_matrix = (noise_std**2) * (
        (1.0 - rho) * identity_matrix + rho * ones_matrix
    )
    return noise_covariance_matrix


def generate_factor_C_var1(
    time_count: int,
    rank_dimension: int,
    ar_param: float,
    noise_std: float,
    seasonal_strength: float = 0.5,
    period: float = 4.0,
    rho_when_ar_positive: float = 0.35,
) -> np.ndarray:
    """
    Generate temporal factor C using VAR(1) + seasonality + correlated noise:

        c_t = Phi c_{t-1} + s_t + eps_t

    Shapes:
        factor_C: (T, R)
        Phi:      (R, R)
        s_t:      (R,)
        eps_t:    (R,)

    Returns:
        factor_C: np.ndarray of shape (time_count, rank_dimension)
    """
    factor_C = np.zeros((time_count, rank_dimension), dtype=float)

    var1_transition_matrix = _make_stable_var1_matrix(
        rank_dimension=rank_dimension,
        ar_strength=ar_param,
    )  # (R, R)

    rho = rho_when_ar_positive if ar_param > 0.0 else 0.0

    noise_covariance_matrix = _make_equicorr_cov(
        rank_dimension=rank_dimension,
        noise_std=noise_std,
        rho=rho,
    )  # (R, R)

    noise_cholesky = np.linalg.cholesky(
        noise_covariance_matrix + 1e-10 * np.eye(rank_dimension)
    )  # (R, R)

    seasonal_amp_sin = (
        np.random.uniform(0.4, 1.0, size=(rank_dimension,)) * seasonal_strength
    )  # (R,)
    seasonal_amp_cos = (
        np.random.uniform(0.0, 0.6, size=(rank_dimension,)) * seasonal_strength
    )  # (R,)
    seasonal_phase_sin = np.random.uniform(
        0.0, 2.0 * np.pi, size=(rank_dimension,)
    )  # (R,)
    seasonal_phase_cos = np.random.uniform(
        0.0, 2.0 * np.pi, size=(rank_dimension,)
    )  # (R,)

    # c_0 ~ N(0, I)
    factor_C[0, :] = np.random.normal(1.0, 1.0, size=(rank_dimension,))

    for time_index in range(1, time_count):
        if ar_param > 0.0:
            omega1 = 2.0 * np.pi * time_index / period
            omega2 = 2.0 * np.pi * (2.0 * time_index) / period
            seasonal_vector = seasonal_amp_sin * np.sin(
                omega1 + seasonal_phase_sin
            ) + seasonal_amp_cos * np.cos(
                omega2 + seasonal_phase_cos
            )  # (R,)
        else:
            seasonal_vector = np.zeros((rank_dimension,), dtype=float)

        standard_normal_vector = np.random.normal(
            0.0, 1.0, size=(rank_dimension,)
        )  # (R,)
        noise_vector = noise_cholesky @ standard_normal_vector  # (R,)

        factor_C[time_index, :] = (
            (var1_transition_matrix @ factor_C[time_index - 1, :])
            + seasonal_vector
            + noise_vector
        )

    return factor_C


def generate_verification_data_relu(
    filename_prefix: str,
    rank_dimension: int,
    ar_param: float,
    noise_std: float,
    observation_noise_ratio: float,
    relu_standardize_before: bool = True,
) -> None:
    """
    Generate a synthetic tensor with a NONLINEAR observation model:

    Step 1 (latent CP signal):
        S[i, j, t] = sum_{r=1..R} A[i,r] * B[j,r] * C[t,r]

        if without 2 - 5, cp perform >= our tensor reconstruction.

    Step 2 (optional standardize):
        S_tilde = S / std(S)   (helps keep scale stable)

    Step 3 (nonlinear link g = ReLU):
        Y = ReLU(S_tilde) = max(0, S_tilde)   element-wise

    Step 4 (observation noise):
        X_raw = Y + E
        std(E) = observation_noise_ratio * std(Y)

    Step 5 (final saved tensor):
        X[t, i, j] = X_raw[i, j, t]   (transpose to [time, firm, feature])

    Notes:
        - We do NOT enforce any "consensus" relationship now.
        - Feature 1 is treated the same as any other feature.
    """

    # 1) CONFIGURATION
    firm_count: int = 150  # N
    feature_count: int = 14  # F
    time_count: int = 150  # T

    seasonal_strength: float = 0.5

    print(f"Generating {filename_prefix} (ReLU nonlinearity)...")
    print(
        f"  -> Config: rank={rank_dimension}, ar_param={ar_param}, noise_std={noise_std}, "
        f"obs_noise_ratio={observation_noise_ratio}, standardize_before_relu={relu_standardize_before}"
    )

    np.random.seed(42)

    # 2) SPATIAL FACTORS A, B
    # A: (N, R), B: (F, R)
    factor_A = np.random.normal(1.0, 1.0, (firm_count, rank_dimension)).astype(float)
    factor_B = np.random.normal(1.0, 1.0, (feature_count, rank_dimension)).astype(float)

    # higher variance N(1, 1.0)

    # 3) TEMPORAL FACTOR C (VAR(1))
    # C: (T, R)
    factor_C = generate_factor_C_var1(
        time_count=time_count,
        rank_dimension=rank_dimension,
        ar_param=ar_param,
        noise_std=noise_std,
        seasonal_strength=seasonal_strength,
        period=40.0,
        rho_when_ar_positive=0.35,
    )

    # 4) CP SIGNAL TENSOR S in shape (N, F, T)
    # TensorLy cp_to_tensor expects factors matching modes: (N,R), (F,R), (T,R)
    weights = np.ones(rank_dimension, dtype=float)  # (R,)
    cp_signal_tensor_nft = tl.cp_to_tensor(
        (weights, [factor_A, factor_B, factor_C])
    )  # (N, F, T)

    # 5) OPTIONAL STANDARDIZATION BEFORE ReLU
    # This keeps scale stable so "observation_noise_ratio" behaves similarly across runs.
    if relu_standardize_before:
        cp_signal_standard_deviation: float = float(np.std(cp_signal_tensor_nft))
        if cp_signal_standard_deviation <= 1e-12:
            cp_signal_standard_deviation = 1.0
        standardized_signal_tensor_nft = (
            cp_signal_tensor_nft - cp_signal_tensor_nft.mean()
        ) / cp_signal_standard_deviation  # (N,F,T)
    else:
        standardized_signal_tensor_nft = cp_signal_tensor_nft  # (N,F,T)

    # 6) NONLINEAR LINK: g = ReLU (element-wise)
    # ReLU(x) = max(0, x)
    nonlinear_signal_tensor_nft = np.maximum(
        standardized_signal_tensor_nft, 0
    )  # (N,F,T)
    # nonlinear_signal_tensor_nft = np.tanh(standardized_signal_tensor_nft)  # (N,F,T)

    # 7) ADD OBSERVATION NOISE AFTER NONLINEARITY
    nonlinear_signal_standard_deviation: float = float(
        np.std(nonlinear_signal_tensor_nft)
    )
    if nonlinear_signal_standard_deviation <= 1e-12:
        nonlinear_signal_standard_deviation = 1.0

    observation_noise_standard_deviation: float = (
        observation_noise_ratio * nonlinear_signal_standard_deviation
    )

    observation_noise_tensor_nft = np.random.normal(
        loc=0.0,
        scale=observation_noise_standard_deviation,
        size=nonlinear_signal_tensor_nft.shape,
    ).astype(
        float
    )  # (N,F,T)

    observed_tensor_nft = (
        nonlinear_signal_tensor_nft + observation_noise_tensor_nft
    )  # (N,F,T)

    # 8) TRANSPOSE TO FINAL SHAPE: (T, N, F)
    # old: (N, F, T) -> new: (T, N, F)
    observed_tensor_tnf = np.transpose(observed_tensor_nft, (2, 0, 1))  # (T,N,F)

    # 9) SAVE ARTIFACTS
    tensor_filename = f"{filename_prefix}_tensor.npy"
    np.save(tensor_filename, observed_tensor_tnf)

    ticker_filename = f"{filename_prefix}_tickers.npy"
    dummy_tickers = np.array([f"FIRM_{i:03d}" for i in range(firm_count)])
    np.save(ticker_filename, dummy_tickers)

    print(f"  -> Saved Tensor:  {tensor_filename} {observed_tensor_tnf.shape}")
    print(f"  -> Saved Tickers: {ticker_filename}")
    print("-" * 30)
    # 10) PLOT DISTRIBUTION (raw and standardized)
    plot_tensor_distribution(
        tensor_values_tnf=observed_tensor_tnf,
        filename_prefix=filename_prefix,
        sample_size=1_000_000,
        bins=200,
    )


def summarize_tensor_values(
    tensor_values: np.ndarray,
    name: str,
) -> None:
    """
    Print simple numeric summary of tensor values.

    tensor_values: np.ndarray, any shape, float
    name: string label for printing
    """
    flat_values: np.ndarray = tensor_values.reshape(-1)

    # remove NaN just in case (your generator has no NaN, but safe)
    finite_values: np.ndarray = flat_values[np.isfinite(flat_values)]

    if finite_values.size == 0:
        print(f"[{name}] No finite values to summarize.")
        return

    mean_value: float = float(np.mean(finite_values))
    std_value: float = float(np.std(finite_values))
    min_value: float = float(np.min(finite_values))
    max_value: float = float(np.max(finite_values))

    p01_value: float = float(np.percentile(finite_values, 1))
    p50_value: float = float(np.percentile(finite_values, 50))
    p99_value: float = float(np.percentile(finite_values, 99))

    fraction_equal_zero: float = float(np.mean(finite_values == 0.0))
    fraction_less_than_zero: float = float(np.mean(finite_values < 0.0))
    fraction_greater_than_zero: float = float(np.mean(finite_values > 0.0))

    print(f"\n[{name}]")
    print(f"count = {finite_values.size}")
    print(f"mean  = {mean_value:.6f}")
    print(f"std   = {std_value:.6f}")
    print(f"min   = {min_value:.6f}")
    print(f"p01   = {p01_value:.6f}")
    print(f"p50   = {p50_value:.6f}")
    print(f"p99   = {p99_value:.6f}")
    print(f"max   = {max_value:.6f}")
    print(f"fraction == 0 = {fraction_equal_zero:.6f}")
    print(f"fraction < 0  = {fraction_less_than_zero:.6f}")
    print(f"fraction > 0  = {fraction_greater_than_zero:.6f}")


def plot_tensor_distribution(
    tensor_values_tnf: np.ndarray,  # shape [T, N, F]
    filename_prefix: str,
    sample_size: int = 1_000_000,
    bins: int = 200,
) -> None:
    """
    Plot tensor value distributions and save PNG files.

    We save:
      1) raw histogram
      2) standardized (z-score) histogram vs N(0,1) idea

    tensor_values_tnf: np.ndarray shape [T, N, F], float
    filename_prefix: used in output filenames
    """
    flat_values: np.ndarray = tensor_values_tnf.reshape(-1)

    # sample to avoid very heavy plotting (tensor is ~20M values)
    rng = np.random.default_rng(42)
    total_count: int = int(flat_values.size)

    chosen_sample_size: int = int(min(sample_size, total_count))
    sampled_indices: np.ndarray = rng.integers(
        low=0, high=total_count, size=chosen_sample_size
    )
    sampled_values_raw: np.ndarray = flat_values[sampled_indices]

    # remove NaN just in case
    sampled_values_raw = sampled_values_raw[np.isfinite(sampled_values_raw)]
    if sampled_values_raw.size == 0:
        print("[plot_tensor_distribution] No finite values to plot.")
        return

    # print numeric summary
    summarize_tensor_values(sampled_values_raw, name=f"{filename_prefix} RAW sample")

    # standardized (z-score): (x - mean) / std
    raw_mean: float = float(np.mean(sampled_values_raw))
    raw_std: float = float(np.std(sampled_values_raw))
    if raw_std <= 1e-12:
        raw_std = 1.0
    sampled_values_standardized: np.ndarray = (sampled_values_raw - raw_mean) / raw_std

    summarize_tensor_values(
        sampled_values_standardized, name=f"{filename_prefix} STANDARDIZED sample"
    )

    # ---------- Plot 1: raw histogram ----------
    plt.figure()
    plt.hist(sampled_values_raw, bins=bins)
    plt.title(f"{filename_prefix}: tensor values (raw) histogram (sample)")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()

    raw_plot_filename: str = f"{filename_prefix}_hist_raw.png"
    plt.savefig(raw_plot_filename, dpi=200)
    plt.close()

    # ---------- Plot 2: standardized histogram ----------
    plt.figure()
    plt.hist(sampled_values_standardized, bins=bins)
    plt.title(f"{filename_prefix}: tensor values (standardized) histogram (sample)")
    plt.xlabel("z-score (mean 0, std 1)")
    plt.ylabel("count")
    plt.tight_layout()

    standardized_plot_filename: str = f"{filename_prefix}_hist_standardized.png"
    plt.savefig(standardized_plot_filename, dpi=200)
    plt.close()

    print(f"[INFO] Saved plots: {raw_plot_filename}, {standardized_plot_filename}")


if __name__ == "__main__":
    # High covariance example (ReLU makes it NONLINEAR, so CP may underfit now)
    generate_verification_data_relu(
        filename_prefix="verify_high_covar_nonlinear",
        rank_dimension=30,
        ar_param=0.9,
        noise_std=0.1,
        observation_noise_ratio=0.5,
        relu_standardize_before=True,
    )

    # No covariance example (still NONLINEAR, but time factor has no memory)
    generate_verification_data_relu(
        filename_prefix="verify_no_covar_nonlinear",
        rank_dimension=100,
        ar_param=0.5,
        noise_std=0.5,
        observation_noise_ratio=0.8,
        relu_standardize_before=True,
    )

    print("Verification data generation complete (ReLU version).")
"""
plot the tensor distribution: standard, N(0, 1)
"""
