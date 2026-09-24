# Tensor-TimesFM: High-dimensional Sparse Time Series Forecasting with Tensor Decoders and Latent Dynamics Prediction

> **ECML PKDD 2026** - Research Track  
> Anonymous Authors

---

## Overview

**Tensor-TimesFM** is an end-to-end framework for forecasting high-dimensional sparse time-series tensors. It couples a nonlinear tensor decoder with a frozen time-series foundation model, **TimesFM**, so that forecasting is performed in a denoised latent factor space rather than directly on sparse, noisy, high-dimensional observations.

The central idea is to decompose a tensor into static structural factors and dynamic temporal factors, forecast the temporal factors with TimesFM, and decode future tensor entries with an MLP-based tensor decoder. This preserves tensor structure, handles missing observations through mask-aware losses, and improves robustness on sparse business and financial forecasting tasks.

![Tensor-TimesFM model architecture](assets/TensorTimesFM_model.svg)

---

## Key Results

Tensor-TimesFM is evaluated on synthetic tensors and three real-world datasets: EPS, Walmart sales, and Rideshare. The results below report mean +/- standard deviation over 10 seeds when available.

| Dataset | Comparison | Baseline RMSE | Baseline R2 | Tensor-TimesFM RMSE | Tensor-TimesFM R2 |
|---|---|---:|---:|---:|---:|
| EPS | Tensor-GRU | 0.8534 +/- 0.0027 | 0.5929 +/- 0.0026 | **0.5534 +/- 0.0032** | **0.8288 +/- 0.0020** |
| EPS | TimesFM | 0.7671 +/- 0.0001 | 0.6711 +/- 0.0001 | **0.5534 +/- 0.0032** | **0.8288 +/- 0.0020** |
| Walmart | TensorCast | 2.7884 +/- 0.0710 | 0.4073 +/- 0.0304 | **2.2787 +/- 0.0493** | **0.5993 +/- 0.0017** |
| Walmart | TimesFM | 2.2904 +/- 0.0023 | **0.6007 +/- 0.0008** | **2.2787 +/- 0.0493** | 0.5993 +/- 0.0017 |
| Rideshare | TimesFM | 2.4568 +/- 0.0043 | 0.9341 +/- 0.0004 | **2.1040 +/- 0.0419** | **0.9475 +/- 0.0021** |

The largest gains appear in sparse settings such as EPS and Rideshare, where tensor-structured latent representations help TimesFM operate on cleaner temporal trajectories.

---

## Contributions

- **Tensor decoder for sparse forecasting.** A nonlinear MLP-based decoder learns latent factor matrices with SGD and computes loss only on observed entries.
- **Latent-space foundation-model forecasting.** TimesFM forecasts temporal latent factors instead of flattened raw observations, improving robustness under sparsity.
- **Mask-aware tensor prediction.** The framework supports sparse reconstruction, multi-step forecasting, and downstream target prediction with regression losses.
- **Empirical validation across domains.** Experiments cover synthetic tensors, retail demand, EPS forecasting, and rideshare demand with missing values and heterogeneous tensor shapes.

---

## Repository Structure

```text
Tensor-TimesFM-ECML-2026/
|
|-- README.md
|-- TensorTimesFM_model.pdf
|-- assets/
|   |-- TensorTimesFM_model.png
|   `-- TensorTimesFM_model.svg
|-- ECML2026datasets.zip
|-- dataset_truncation_description.txt
|-- environment.yml
|
|-- examples/
|   |-- full_command_EPS_best.sh
|   |-- EPS_TimesFM_debug_best.py
|   |-- MC_tensor_generation_Jan15.py
|   |-- cp_reconstruction_Jan16.py
|   |-- run_cpmlp_grid.sh
|   `-- rideshare_3d_tensor/
|       |-- Rideshare_TimesFM_debug_Sept22_3losses_ortho_multi_step.py
|       `-- sweep_Rideshare_TimesFM_Sept23_multi_step.yaml
|
|-- walmart_lagp_sales_sweep_Nov17.yaml
|-- walmart_lagp_TimesFM_debug_Nov_17.py
`-- LICENSE
```

---

## Installation

### Requirements

- Python 3.11
- CUDA-capable NVIDIA GPU recommended
- Conda or Mamba for environment management
- W&B account for sweep-based experiments

### Setup

```bash
git clone git@github.com:LarryC01/Tensor-TimesFM-ECML-2026.git
cd Tensor-TimesFM-ECML-2026
```

Create and activate the environment:

```bash
conda create -n tensorts python=3.11 "pip>=24"
conda activate tensorts
```

Install the PyTorch + CUDA runtime:

```bash
conda install -c nvidia -c pytorch pytorch pytorch-mutex=*=cuda pytorch-cuda=11.8 cuda-nvcc
```

Install the scientific Python stack:

```bash
conda install -c conda-forge numpy scipy pandas scikit-learn scikit-image einops sympy sentencepiece
```

Install sparse tensor support:

```bash
conda install -c pyg -c pytorch -c conda-forge pytorch-sparse~=0.6.18
```

Install the Hugging Face and forecasting stack:

```bash
conda install -c conda-forge transformers datasets accelerate evaluate "tokenizers>=0.13.1" peft sktime
```

Install Lightning and metrics:

```bash
conda install -c conda-forge lightning~=2.2.5 torchmetrics
```

Install experiment, visualization, and utility packages:

```bash
conda install -c conda-forge psutil nvtop tqdm rich attrs tabulate typing_extensions absl-py python-dotenv ruamel.yaml omegaconf typed-argument-parser hydra-core hydra-submitit-launcher jsonargparse "typeshed-client>=2.1.0" openpyxl pandarallel patool torchinfo plotly matplotlib seaborn moviepy jupyter wandb tensorboardx click setuptools_scm twine blackd mypy bump2version watchdog flake8 tox coverage sphinx
```

Install pip-only packages:

```bash
pip install bitsandbytes~=0.43.1
pip install build datargs sourcery torchopt torch_optimizer separableconv-torch cd2root omegaconf-argparse cookiecutter reformer_pytorch
pip install black "black[jupyter]"
```

Optional, only if needed for your GPU setup:

```bash
pip install flash-attn
```

Log in to W&B before running sweep experiments:

```bash
wandb login
```

---

## Data

### Dataset

The release includes `ECML2026datasets.zip`, a compact package that preserves the structure of the experimental tensors while protecting large or proprietary data.

| File in package | Format | Original shape | Released handling |
|---|---|---:|---|
| `rideshare_tensor.npy` | NumPy array | `(541, 156, 15)` | Copied without changes |
| `tensor_1_units.pt` | PyTorch tensor | `(1941, 3049, 10)` | Truncated to 10% along axis 0 |
| `tensor_2_price.pt` | PyTorch tensor | `(1941, 3049, 10)` | Truncated to 10% along axis 0 |
| `tensor_3_event.pt` | PyTorch tensor | `(1941, 41)` | Copied without changes |
| `fundamentals_analyst_forecast_EPS_regression.npy` | NumPy array | `(158, 9583, 14)` | Truncated to 10% along axis 0 |

The packaging logic is documented in `dataset_truncation_description.txt`. The full EPS tensor is proprietary and requires institutional WRDS access.

### Real-world Dataset Statistics

| Dataset | Shape | Total entries | Non-missing ratio | Zero-value ratio | Forecast horizon |
|---|---:|---:|---:|---:|---:|
| EPS | `(158, 9583, 14)` | 21,197,596 | 18.99% | 3.66% | 1 |
| Walmart Sales Tensor | `(1941, 3049, 10)` | 59,181,090 | 100.00% | 68.00% | 28 |
| Walmart Prices Tensor | `(1941, 3049, 10)` | 59,181,090 | 79.22% | 0.00% | 28 |
| Walmart Event Matrix | `(1941, 41)` | 79,581 | 100.00% | 97.15% | 28 |
| Rideshare | `(541, 156, 15)` | 1,265,940 | 59.39% | 5.89% | 168 |

---

## Training

### Walmart Experiment

The Walmart experiment uses a W&B sweep file and the corresponding TimesFM training script:

```bash
wandb sweep ./walmart_lagp_sales_sweep_Nov17.yaml
```

The sweep launches:

```text
./walmart_lagp_TimesFM_debug_Nov_17.py
```

After W&B creates a sweep ID, start an agent with the command printed by W&B:

```bash
wandb agent <entity>/<project>/<sweep_id>
```

### EPS Experiment

The released EPS data contains only the shared 10% slice for data-protection reasons. The full EPS benchmark uses proprietary WRDS data.

Run from the `examples/` directory:

```bash
cd examples
bash full_command_EPS_best.sh
```

The shell helper launches:

```text
examples/EPS_TimesFM_debug_best.py
```

The script runs the selected EPS Tensor-TimesFM configuration across multiple seeds and summarizes `Forecasted_R2` into CSV files under the configured output directory.

### Rideshare Experiment

The Rideshare multi-step experiment uses:

```text
examples/rideshare_3d_tensor/Rideshare_TimesFM_debug_Sept22_3losses_ortho_multi_step.py
```

W&B sweep file:

```text
examples/rideshare_3d_tensor/sweep_Rideshare_TimesFM_Sept23_multi_step.yaml
```

Launch:

```bash
wandb sweep examples/rideshare_3d_tensor/sweep_Rideshare_TimesFM_Sept23_multi_step.yaml
wandb agent <entity>/<project>/<sweep_id>
```

### Synthetic Experiment

The synthetic Monte Carlo workflow from the original README is preserved below.

#### Step 1: Monte Carlo Tensor Generation

Open:

```text
examples/MC_tensor_generation_Jan15.py
```

In the configuration section near the top, choose a tensor size appropriate for your device. Make sure a random seed is set for reproducibility.

Run from the repository root:

```bash
time python examples/MC_tensor_generation_Jan15.py
```

After successful execution, the following files are generated in the repository root:

```text
verify_high_covar_nonlinear.npy
verify_no_covar_nonlinear.npy
```

These correspond to high-covariance and no/low-covariance synthetic tensor settings.

#### Step 2: CP Reconstruction Baseline

This step runs standard CP decomposition using only the first 80% of time steps.

High-covariance example:

```bash
time python examples/cp_reconstruction_Jan16.py \
  --tensor_path ./verify_high_covar_nonlinear.npy \
  --cp_rank 30 \
  --seed 123
```

No/low-covariance example:

```bash
time python examples/cp_reconstruction_Jan16.py \
  --tensor_path ./verify_no_covar_nonlinear.npy \
  --cp_rank 100 \
  --seed 123
```

Each run appends results to:

```text
results/cp_optionB_results.csv
```

#### Step 3: Tensor Reconstruction with CP-MLP

Run:

```bash
time bash examples/run_cpmlp_grid.sh
```

The default settings were tuned for the large tensor `(150, 9500, 14)` and may need smaller `epochs` or `batch_size` for local debugging.

Each run appends results to:

```text
results/cp_mlp_optionB_results.csv
```

#### Synthetic Sanity Checks

- The same seed should produce identical or nearly identical results.
- High-covariance tensors should reconstruct better than no/low-covariance tensors.
- CP and CP-MLP should use the same 80% in-sample split.
- CSV files should append results after each run.

---

## Reproducibility

### Checklist

- [x] Code for Tensor-TimesFM, CP-TimesFM, Tensor-GRU, TimeGrad, and TimesFM baselines is included.
- [x] Public or partially shared dataset package is provided in `ECML2026datasets.zip`.
- [x] EPS truncation and data-packaging logic is documented.
- [x] Synthetic tensor generation and reconstruction workflows are included.
- [x] W&B sweep files are provided for Walmart and Rideshare.
- [x] Main reported metrics include RMSE and R2.

### Known Reproducibility Limitations

- The full EPS dataset is proprietary and cannot be released. Results on the full 21,197,596-entry EPS dataset require institutional access to WRDS.
- Training scripts may contain absolute file paths specific to the original compute cluster. These must be updated before running.
- Exact numerical results may vary slightly across hardware configurations due to floating-point non-determinism in GPU operations.

### Random Seed

Set the following before training for reproducibility:

```python
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## Citation

If you use this repository or build on this work, please cite:

```bibtex
@inproceedings{tensortimesfm2026,
  title     = {Tensor-TimesFM: High-dimensional Sparse Time Series Forecasting with Tensor Decoders and Latent Dynamics Prediction},
  author    = {Anonymous},
  booktitle = {ECML PKDD},
  year      = {2026}
}
```

---

## Related Work

This project builds on and connects to:

- **TimesFM** - A decoder-only foundation model for zero-shot time-series forecasting.
- **TensorCast** - Coupled tensor factorization for contextual forecasting.
- **TimeGrad** - Autoregressive diffusion modeling for probabilistic time-series forecasting.
- **TRMF and tensor autoregressive factorization** - Low-rank temporal factor models for missing-value forecasting.
- **DeepGLO** - Global-local deep forecasting with low-dimensional shared representations.
- **Classical tensor decomposition** - CP and Tucker decomposition methods for structured multi-way data.

Tensor-TimesFM differs from standard foundation-model forecasting by explicitly preserving tensor modes and forecasting learned latent temporal factors instead of flattened raw observations.

---

## License

This project is released for research purposes. The code is provided under the [MIT License](LICENSE). Some included or adapted components may be governed by their own licenses, including [LICENSE.gluon-ts](LICENSE.gluon-ts). Datasets remain subject to their original providers' licensing terms, including WRDS/I/B/E/S, Kaggle M5 Walmart data, and the Rideshare source dataset.
