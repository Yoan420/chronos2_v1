#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MLP_SCRIPT="${MLP_SCRIPT:-./examples/our_tensor_reconstruction_Jan16.py}"
OUT_CSV="${OUT_CSV:-./results/cp_mlp_raw_results.csv}"

time $PYTHON $MLP_SCRIPT \
    --tensor_npy_path "verify_high_covar_nonlinear_tensor.npy" \
    -r 30 \
    -e 100 \
    --batch_size 65536 \
    --output_csv "$OUT_CSV" \
    --seed 1881


time $PYTHON $MLP_SCRIPT \
    --tensor_npy_path "verify_no_covar_nonlinear_tensor.npy" \
    -r 100 \
    -e 100 \
    --batch_size 65536 \
    --output_csv "$OUT_CSV" \
    --seed 1881

echo "[DONE] CP-MLP results appended to: $OUT_CSV"
